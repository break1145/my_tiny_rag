import json

from OpenaiModel import OpenAIModel


def split_query(query):
    model = OpenAIModel()
    prompt = f"""
        I hope you can describe this question from different angles so that I can perform vector matching from multiple 
        perspectives. I hope you can generate multiple sub-questions based on this question, or abstract this question 
        using another text. However, you must not fabricate information; the generation must be strictly based on this question.
        The generated content must strictly follow the following format, separated by '\n'.
        Content 1:xxx
        Content 2:xxx
        Here is the original query:{query}
    """
    res = model.get_response(prompt)
    print(res)
    queries = json.loads(res).get("choices")[0].get('message').get('content').split('\n')
    queries.append(query)
    return queries

# split_query("Did Lincoln sign the National Banking Act of 1863?")


# try to use function call

import json
# 1. 定义 Function Schema
function_schema = {
  "name": "generate_query_variations",
  "description": "Generates multiple sub-queries or variations of the original query for vector matching.",
  "parameters": {
    "type": "object",
    "properties": {
      "variations": {
        "type": "array",
        "description": "A list of alternative or sub-queries derived from the original query.",
        "items": {
          "type": "string",
          "description": "An alternative query string."
        }
      }
    },
    "required": ["variations"]
  }
}

def split_query_with_function_call(query):
    # 2. 构建发送给 API 的消息和函数定义
    messages = [
        {"role": "user", "content": f"""
        Based on the following original query, generate multiple alternative sub-queries or abstract reformulations 
        from different angles for vector matching. Ensure the generated content is strictly based on the original query 
        and does not fabricate information.
        And use the same language as the original query to reply.

        Original query: {query}
        """}
    ]

    # 3. 调用 LLM API，并提供 function_schema
    model = OpenAIModel()
    response = model.get_response_with_fc(
        prompt="",
        tools=[{"type": "function", "function": function_schema}],
        tool_choice="auto",
        messages=messages,
    )

    # 4. 解析 API 响应
    generated_queries = []
    response_message = json.loads(response)["choices"][0]["message"]

    if response_message.get("tool_calls"):
        tool_call = response_message["tool_calls"][0]
        if tool_call["function"]["name"] == "generate_query_variations":
            function_args = json.loads(tool_call["function"]["arguments"])
            generated_queries = function_args.get("variations", [])

    generated_queries.append(query)
    return generated_queries

#
# original_query = "丁真的电子烟和王源的传统烟比起来哪个吸了肺痒痒？"
# result_queries = split_query_with_function_call(original_query)
# [print(x) for x in result_queries]