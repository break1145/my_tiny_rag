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