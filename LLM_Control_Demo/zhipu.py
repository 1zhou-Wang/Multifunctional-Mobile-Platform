from zhipuai import ZhipuAI

import audio

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_item",
            "description": "根据文字信息输出需要抓取的物品，从'bowl' 和 'cup' 中选择一个",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "description": "需要抓取的物品",
                        "type": "string"
                    }
                },
                "required": [ "item" ]
            },
        }
    },
] 



client = ZhipuAI(api_key="***") 
messages = []
messages.append({"role": "user", "content": "帮我查询从2024年1月20日，从北京出发前往上海的航班"})
response = client.chat.completions.create(
    model="glm-4",  
    messages=messages,
    tools=tools,
)
print(response.choices[0].message)
messages.append(response.choices[0].message.model_dump()) 