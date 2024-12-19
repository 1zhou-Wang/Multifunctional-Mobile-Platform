# %%
import warnings

warnings.filterwarnings("ignore")
from airbot.backend import Arm, Camera, Base, Gripper
import os
import numpy as np
import copy
from airbot.backend.utils.utils import camera2base, armbase2world
from airbot.lm import Detector, Segmentor
from airbot.grasp.graspmodel import GraspPredictor
from PIL import Image
import time
import cv2
from airbot.example.utils.draw import draw_bbox, obb2poly
from airbot.example.utils.vis_depth import vis_image_and_depth
from scipy.spatial.transform import Rotation
from threading import Thread, Lock
import math
import torch
import rospy

from typing import Tuple
from airbot.lm.utils import depth2cloud

os.environ['LM_CONFIG'] = "/root/Workspace/AXS_baseline/ICRA2024-Sim2Real-AXS/local.yaml"
os.environ['CKPT_DIR'] = '/root/Workspace/AXS_baseline/ckpt'

CAR_2_ARMBASE = np.array([0.2975, -0.17309, 0.3488])
CAMERA_SHIFT = np.array([-0.093, 0, 0.07]), np.array([0.5, -0.5, 0.5, -0.5])
YOLO_LABEL = ["mug", "bowl", "cabinet", "cabinet_handle", "cabinet_middle", "microwave", "microwave_door"]

CHECK_BOWL_FROM_FRONT = np.array([0.0, -0.4, 0.0, 0.0, 0.0, 0.0])
INIT_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


POS1 = np.array([0.0009536888683214784, -1.9155794382095337, 1.4158464670181274, -0.3603036403656006, 0.086785688996315, 0.2958342730998993])
P_1 = np.array([0.050164032727479935, -1.1957350969314575, 2.188715934753418, -0.36755168437957764, -0.02994582988321781, 0.2576867341995239])
P_2 = np.array([1.9892042875289917, -1.1160067319869995, 2.188715934753418, -0.3614480793476105, 0.11615930497646332, 0.2653162479400635])
P_3 = np.array([1.8492027521133423, -1.1388952732086182, 1.0248340368270874, -0.32940414547920227, -0.04558632895350456, 0.2775234580039978])
P_4 = np.array([1.7183566093444824, -1.4215686321258545, 1.2308309078216553, 0.0036240178160369396, -1.5642404556274414, 0.31681543588638306])
P_5 = np.array([1.640535593032837, -1.574158787727356, 1.4353017807006836, -0.20275425910949707, -1.8255512714385986, 0.34809643030166626])
P_6 = np.array([1.1312657594680786, -1.2106126546859741, 1.1900129318237305, -0.020027466118335724, -1.509689450263977, 0.3736552894115448])
P_7 = np.array([0.0009536888683214784, -1.9155794382095337, 1.4158464670181274, -0.3603036403656006, 0.086785688996315, 0.2958342730998993])
P_8 = np.array([0.0009536888683214784, -1.9155794382095337, 1.4158464670181274, -0.3603036403656006, 0.086785688996315, 0.2958342730998993])
POS2 = np.array([0.0009536888683214784, -1.9155794382095337, 1.4158464670181274, -0.3603036403656006, 0.086785688996315, 0.2958342730998993])

CHECK_BOWL_FROM_HEAD_V = (np.array([0.28672, -0.0025376, 0.26345]), np.array([0.00020779, 0.58108, 0.00032349, 0.81385]))

class Solution:
    def __init__(self):
        self.cnt = 0

        self.camera = Camera(backend='ros')
        print("-------------init camera success")

        self.detector = Detector(model='yolo-v8')
        print("-------------init yolo8 success")

        self.base = Base(backend='ros')
        print("-------------init base success")

        self.arm = Arm(backend='ros')
        print("-------------init arm success")

        self.gripper = Gripper(backend='ros')
        print("-------------init Gripper success")

        self.start_check = False
        self.start_vis = False
        self._prompt = 'bowl'
        self.update_once()
        self.t_vis = Thread(target=self.vis, daemon=True)
        self.t_vis.start()
        self.t_update = Thread(target=self.update, daemon=True)
        self.t_update.start()

    def get_image(self):
        return copy.deepcopy(self._image)
    
    def save_image(self):
        img=self.get_image()
        cv2.imwrite("img" + ".bmp", img)
        
    def get_depth(self):
        return copy.deepcopy(self._depth)
    def get_bbox(self):
        return copy.deepcopy(self._bbox)
    def get_sorce(self):
        return copy.deepcopy(self._score)

    def update(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.start_check == True:
                self.update_once()
            rate.sleep()

    def update_once(self):
        image = self.camera.get_rgb()
        self._image = copy.deepcopy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._depth = copy.deepcopy(self.camera.get_depth())
        det_result = self.detector.infer(self._image, self._prompt)
        self._score = det_result['score']
        self._bbox = det_result['bbox']#.numpy().astype(int)

    def vis(self):
        try:
            while not rospy.is_shutdown():
                if self.start_vis == True:
                    self.vis_once(False)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("Exiting due to user interruption.")
        finally:
            cv2.destroyAllWindows()

    def vis_once(self, write_flag, write_bbox_flag = False):
        get_img = self.get_image()
        get_sorce = self.get_sorce()
        get_bbox = self.get_bbox()

        if write_flag == True:
            #cv2.imwrite(str(self.cnt) + self._prompt + ".bmp", self.image)
            if self.cnt < 10:
                cv2.imwrite("img00" + str(self.cnt) + ".bmp", get_img)
            elif self.cnt < 100:
                cv2.imwrite("img0" + str(self.cnt) + ".bmp", get_img)
            else:
                cv2.imwrite("img" + str(self.cnt) + ".bmp", get_img)

            if write_bbox_flag == False:
                self.cnt = self.cnt + 1

        np_bbox = np.array(get_bbox)
        image_draw = draw_bbox(get_img, obb2poly(np_bbox[None, ...]).astype(int))
        image_show = image_draw.astype(np.uint8)
        cv2.putText(image_show, f"det score: {get_sorce}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_show, f"box: {get_bbox}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_show, f"prompt: {self._prompt}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if write_flag == True and write_bbox_flag == True:
            if self.cnt < 10:
                cv2.imwrite("img00" + str(self.cnt) + self._prompt + ".bmp", image_show)
            elif self.cnt < 100:
                cv2.imwrite("img0" + str(self.cnt) + self._prompt + ".bmp", image_show)
            else:
                cv2.imwrite("img" + str(self.cnt) + self._prompt + ".bmp", image_show)
            self.cnt = self.cnt + 1
        else:
            cv2.imshow('RGB', image_show)

    @staticmethod
    def _bbox2mask(image, bbox):
        mask = np.zeros_like(image[:, :, 0], dtype=bool)
        mask[
            bbox[0] - bbox[2] // 2:bbox[0] + bbox[2] // 2,
            bbox[1] - bbox[3] // 2:bbox[1] + bbox[3] // 2,
        ] = True
        return mask

    @staticmethod
    def base_cloud(image, depth, intrinsic, shift, end_pose):
        cam_cloud = depth2cloud(depth, intrinsic)
        cam_cloud = np.copy(np.concatenate((cam_cloud, image), axis=2))
        return camera2base(cam_cloud, shift, end_pose)

    @staticmethod
    def _vis_grasp(cloud):
        import open3d as o3d
        p = cloud[:, :, :3].reshape(-1, 3).astype(np.float32)
        p_reshaped = p.reshape(92160, 10, 3)
        p = p_reshaped[:, 0, :]
        print(p.shape)

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(p)
        o3d.visualization.draw_geometries([o3d_cloud])

    def testdepth2cloud(self, depth_im, intrinsic_mat, bbox):
        height, width = depth_im.shape
        fx, fy, cx, cy = intrinsic_mat[0][0], intrinsic_mat[1][1], intrinsic_mat[0][2], intrinsic_mat[1][2]
        assert (depth_im.shape[0] == height and depth_im.shape[1] == width)
        xmap = np.arange(width)
        ymap = np.arange(height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        
        x = bbox[0] // 1
        y = bbox[1] // 1
        points_z = depth_im  # change the unit to metel
        points_x = (xmap - cx) * points_z / fx
        points_y = (ymap - cy) * points_z / fy
        print(f"points_z = {points_z[x, y]}")
        print(f"{points_x[x, y]} = ({xmap[x, y]} - {cx}) * {points_z[x, y]} / {fx} ")
        print(f"{points_y[x, y]} = ({ymap[x, y]} - {cy}) * {points_z[x, y]} / {fy} ")
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        return cloud[x, y]


    def lookforonce(self, det_th, frame="armbase"):
        """
            return the center point in different frame like camera, armbase, world
        """

        get_rgb = self.get_image()
        get_depth = self.get_depth()
        # use matplot to show the image and depthr
        #print("showing the image and depth")
        #vis_image_and_depth(get_rgb, get_depth)
        
        get_score = self.get_sorce()
        get_bbox = self.get_bbox()
            
        print(self._prompt + "det score:", get_score)
        #print(self._prompt + "sam score:", _sam_result['score'])
        if get_score > det_th :
            print(f"Found the {self._prompt}")
            # centerpoint from depth image to camera frame
            centerpoint = depth2cloud(get_depth, self.camera.INTRINSIC, organized=True)[get_bbox[0] // 1, get_bbox[1] // 1]
            #centerpoint = self.testdepth2cloud(get_depth, self.camera.INTRINSIC, get_bbox)
            print(f"the first center point = {centerpoint} ")
            # centerpoint from camera frame to armbase frame
            if frame == "armbase" or frame == "world":            
                centerpoint1 = camera2base(centerpoint, CAMERA_SHIFT, self.arm.end_pose)
                print(f"ceneter point = {centerpoint1}")
            
        
            #cloud = self.base_cloud(get_rgb, get_depth, self.camera.INTRINSIC, CAMERA_SHIFT, self.arm.end_pose)
            #self._vis_grasp(cloud)
        
            # centerpoint from armbase frame to world frame
            #if frame == "world":
            #    centerpoint = (armbase2world(centerpoint, (self.base.position, self.base.rotation)).squeeze())
            object_rgb = get_rgb[get_bbox[0] - np.int32(get_bbox[2]/4):get_bbox[0] + np.int32(get_bbox[2]/4), get_bbox[1] - np.int32(get_bbox[3]/4):get_bbox[1] + np.int32(get_bbox[3]/4)]
            mean_rgb = (np.mean(np.mean(object_rgb, axis=0), axis=0).astype(int))
            print(f"----------------- centerpoint in {frame} frame is {centerpoint} and rgb is {mean_rgb} --------------")
            return centerpoint1, mean_rgb
        else:
            return None, None

    def handle_arm_move(self, tar_pos, tar_rot):
        dis = np.linalg.norm(tar_pos)#base on arm frame
        print(f"-------tarpos={tar_pos}, dis={dis}")
        ret = self.arm.move_end_to_pose2(tar_pos, tar_rot)
        return ret

    def grasp(self, cp):
        grasp_position = cp
        grasp_position[0] = cp[0]
        grasp_position[2] = cp[2] + 0.015
        #grasp_rotation = np.array(Rotation.from_euler('xyz', [-179, 0, -179], degrees=True).as_quat())
        grasp_rotation = Rotation.from_euler('xyz', [0, np.pi / 2, np.pi / 2], degrees=False).as_quat()
        
        self.handle_arm_move(grasp_position, grasp_rotation)
        time.sleep(0.5)
        self.gripper.close()
        time.sleep(3)
        self.arm.move_joint_to_pose(INIT_POS)
        time.sleep(2)
        
    def place_c(self, cp):
    # 设置放置位置
        place_position = cp
        place_position[0] = cp[0]
        place_position[2] = cp[2] + 0.015  # 调整高度

    # 设置放置姿态,与抓取姿态相同
        place_rotation = Rotation.from_euler('xyz', [0, np.pi / 2, np.pi / 2], degrees=False).as_quat()

    # 移动机械臂到放置位置
        self.handle_arm_move(place_position, place_rotation)
        time.sleep(0.5)

    # 打开夹爪
        self.gripper.open()
        time.sleep(3)  # 等待夹爪完全打开

    # 将机械臂移回初始位置
        self.arm.move_joint_to_pose(INIT_POS)
        time.sleep(2)

    
    def test(self):
        #"""
        cmd = self.arm.get_joint_pose()
        setcmd = [cmd[0],cmd[1],cmd[2],cmd[3],cmd[4],cmd[5]]
        #有时候我们可能只需要使用机械臂的部分关节位置信息,而不是全部 6 个关节,因此将数据分散到 setcmd 列表中方便后续的操作。
        while True:
            #print("[LANDMARK] ===== car_base_link position:" + str(self.base._position) + ", rot is " + str(self.base._cur_yaw))
            print("[LANDMARK] ===== eef position:          " + str(self.arm.end_pose[0]) + ", rot is " + str(self.arm.quat2euler(self.arm.end_pose[1], True)))
            #end-effector position and orientation
            print("joint is " + str(self.arm.get_joint_pose()))
            inn = input("control")
            if inn == "1":
                setcmd[0] = setcmd[0] - 0.01
            elif inn == "2":
                setcmd[0] = setcmd[0] + 0.01
            elif inn == "3":
                setcmd[1] = setcmd[1] - 0.01
            elif inn == "4":
                setcmd[1] = setcmd[1] + 0.01
            elif inn == "5":
                setcmd[2] = setcmd[2] - 0.01
            elif inn == "6":
                setcmd[2] = setcmd[2] + 0.01
            elif inn == "7":
                setcmd[3] = setcmd[3] - 0.01
            elif inn == "8":
                setcmd[3] = setcmd[3] + 0.01
            elif inn == "9":
                setcmd[4] = setcmd[4] - 0.01
            elif inn == "0":
                setcmd[4] = setcmd[4] + 0.01
            elif inn == "-":
                setcmd[5] = setcmd[5] - 0.01
            elif inn == "=":
                setcmd[5] = setcmd[5] + 0.01
            elif inn == 'c':
                self.update_once()
                self.vis_once(True)
            elif len(inn) > 2:
                self._prompt = inn
                continue
            else:
                continue

            self.arm.move_joint_to_pose(setcmd, True)
        #"""
        while True:
            inn = input("control")
            print("log:" + str(self.cnt))
            self.update_once()
            self.vis_once(True)

def start_grasp(bowl_colour):
    print("-------------start")
    s = Solution()
    
    # s.base.cmd_vel_target_asyn([0.2, 0.0, 0.0], 5.0)
    # print("move forward finish")
    # time.sleep(1)
    # s.base.cmd_vel_target_asyn([0.0, 0.2, 0.0], 5.0)
    # print("move right finish")
    # time.sleep(1)
    
    # s.gripper.close()
    # time.sleep(3)
    
    # s.gripper.open()
    # time.sleep(3)
    
    s.handle_arm_move(*CHECK_BOWL_FROM_HEAD_V)
    time.sleep(1)

    # s._prompt = "bowl"
    # s.update_once()
    # s.vis_once(True, True)
    # cp, rgb = s.lookforonce(0.8, "armbase")
    # if cp is not None:
    #     s.grasp(cp)
    
    s._prompt = "cabine_middle"
    s.update_once()
    s.vis_once(True, True)
    cp, rgb = s.lookforonce(0.5, "armbase")
    if cp is not None:
        s.place_c(cp)
        
        # s.arm.move_joint_to_pose(POS1)
        # time.sleep(1)
        # s.gripper.open()
        # time.sleep(1)
        
        # s.arm.move_joint_to_pose(INIT_POS)
        # time.sleep(1)
        # s.arm.move_joint_to_pose(P_1)
        # time.sleep(1)
        
        # s.arm.move_joint_to_pose(P_2)
        # time.sleep(1)
        
        # s.arm.move_joint_to_pose(P_3)
        # time.sleep(1)
        
        # s.arm.move_joint_to_pose(P_4)
        # time.sleep(1)
        
        # s.arm.move_joint_to_pose(P_5)
        # time.sleep(1)
        # s.arm.move_joint_to_pose(P_6)
        # time.sleep(1)
        # s.arm.move_joint_to_pose(INIT_POS)
        
        
        # else:
        #     print("-------none find .end .")



