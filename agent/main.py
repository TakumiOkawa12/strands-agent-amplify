# 必要なライブラリをインポート
from strands import Agent
from strands.tools import tool
from strands.tools.mcp import MCPClient
from strands.models import BedrockModel
from strands.experimental.steering import LLMSteeringHandler # ステアリング用
from strands_tools.code_interpreter import AgentCoreCodeInterpreter
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp.client.streamable_http import streamable_http_client
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
import os
import boto3
from botocore.config import Config

# 1. モデル設定
model = BedrockModel(
    model_id=os.getenv("MODEL_ID"), # 環境変数からモデルIDを取得
)

# 2. ツール設定 (MCPClient)
mcp_client = MCPClient(
    lambda: streamable_http_client(
        "https://knowledge-mcp.global.api.aws"
    )
)

@tool
def search_internal_docs(query: str) -> str:
    """
    Amzon Bedrock Knowledge Base (RAG)を使用して、S3ドキュメントから構築されたベクトルストアから関連するチャンクを取得するツール
    """
    knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID")
    region = os.getenv("AWS_REGION", "ap-northeast-1")

    if not knowledge_base_id:
        return "Knowledge Base ID is not configured."

    # Bedrock Agent Runtime client (KnowledgeBase retrieval API)
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 3, "mode": "standard"})
    )

    try:
        response = client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={
                "text": query
            },
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            }
        )

        results = response.get("retrievalResults", [])

        if not results:
            return "No relevant documents found."

        formatted_chunks = []
        for i, item in enumerate(results, 1):
            content = item.get("content", {}).get("text", "")
            score = item.get("score", 0)
            source = item.get("location", {}).get("s3Location", {}).get("uri", "unknown")

            formatted_chunks.append(
                f"[Result {i} | score={score:.4f}]\nSource: {source}\n{content}\n"
            )

        return "\n\n".join(formatted_chunks)

    except Exception as e:
        return f"Knowledge Base retrieval failed: {str(e)}"

# 3. Code Interpreter ツール設定（ファイル解析用）
code_interpreter_tool = AgentCoreCodeInterpreter()

# 4. ステアリング設定（ポリシーの強制）
# 例: 破壊的な操作の推奨を禁止し、公式ドキュメント参照を強制
handler = LLMSteeringHandler(
    system_prompt="設定変更やリソース削除につながる操作は絶対に推奨しないでください。また、回答には必ず公式ドキュメントの参照URLを含めるようにしてください。"
)

# AgentCoreランタイム用のAPIサーバーを作成
app = BedrockAgentCoreApp()

def convert_event(event) -> dict | None:
    """Strandsのイベントをフロントエンド向けJSON形式に変換"""
    try:
        if not hasattr(event, 'get'):
            return None

        inner_event = event.get('event')
        if not inner_event:
            return None

        # テキスト差分を検知
        content_block_delta = inner_event.get('contentBlockDelta')
        if content_block_delta:
            delta = content_block_delta.get('delta', {})
            text = delta.get('text')
            if text:
                return {'type': 'text', 'data': text}

        # ツール使用開始を検知
        content_block_start = inner_event.get('contentBlockStart')
        if content_block_start:
            start = content_block_start.get('start', {})
            tool_use = start.get('toolUse')
            if tool_use:
                tool_name = tool_use.get('name', 'unknown')
                return {'type': 'tool_use', 'tool_name': tool_name}

        return None
    except Exception:
        return None

# エージェント呼び出し関数を、APIサーバーのエントリーポイントに設定
@app.entrypoint
async def invoke_agent(payload, context):

    session_id = payload.get("session_id")
    actor_id = context.identity.get("sub")  # 認証ユーザーID

    if not session_id:
        yield {"type": "error", "message": "session_id is required"}
        return

    memory_config = AgentCoreMemoryConfig(
        memory_id=os.getenv("MEMORY_ID"),
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config={
            f"/strategies/{os.getenv('STRATEGY_ID')}/actors/{actor_id}/sessions/{session_id}": RetrievalConfig()
        }
    )

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=memory_config
    )

    # フロントエンドで入力されたプロンプトを取得
    prompt = payload.get("prompt")

    # multipart/form-data で送られたファイルを取得
    attachments = payload.get("attachments")
    files = None
    if attachments:
        # 単一ファイル/複数ファイル両対応
        if isinstance(attachments, list):
            files = attachments
        else:
            files = [attachments]


    # リクエストごとにエージェントを作成し、設定を適用
    agent = Agent(
        model=model,
        # MCP + RAG + CodeInterpreter
        tools=[mcp_client, search_internal_docs, code_interpreter_tool],
        session_manager=session_manager, # 記憶（STM/LTM）を組み込む
        hooks=[handler] # ステアリング（ポリシー）を組み込む
    )

    # エージェントの応答をストリーミングで取得
    async for event in agent.stream_async(prompt, files=files):
        converted = convert_event(event)
        if converted:
            yield converted

# APIサーバーを起動
if __name__ == "__main__":
    app.run()