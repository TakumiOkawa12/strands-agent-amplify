from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke_agent(payload, context):
    yield {"type": "text", "data": "OK"}
# APIサーバーを起動
if __name__ == "__main__":
    app.run()