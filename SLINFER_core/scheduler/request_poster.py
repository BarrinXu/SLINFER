import time

import aiohttp
import json
import asyncio


async def post_request(url, headers, data):
    print(time.time())
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            async for line in response.content:
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line == '\n':
                        continue
                    print(time.time())
                    print(decoded_line)


async def main():
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": '/root/autodl-fs/sllm/Llama-2-7b-chat-hf',
        "prompt": list(range(1, 3073)),
        "min_tokens": 4,
        "max_tokens": 4,
        'request_id': '3-000001',
        "stream": True
    }
    # data = {
    #     'request_info': {
    #         "model_id": 1,
    #         "model_type": 'llama-2-7b',
    #         "request_id": '1-000001',
    #         "input_length": 1024,
    #         'expect_output_length': 128,
    #     }
    #
    # }

    tasks = []
    url = f"http://localhost:8000/v1/completions"
    tasks.append(post_request(url, headers, data))

    responses = await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
