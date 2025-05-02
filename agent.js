import * as dotenv from "dotenv"
dotenv.config()

import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts"

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai"
import { HumanMessage, AIMessage } from "@langchain/core/messages"

import { AgentExecutor, createToolCallingAgent } from "langchain/agents"

// import { TavilySearchResults } from "@langchain/community/tools/tavily_search"
import { createRetrieverTool } from "langchain/tools/retriever"
import { TaskType } from "@google/generative-ai"
// Custom Data Source, Vector Stores
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { TavilySearch } from "@langchain/tavily"
import readline from "readline"
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-pro",
  temperature: 0.5,
  maxRetries: 3,
})

const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are a helful assistant called Jarvis."),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
])
// Search Tool
const searchTool = new TavilySearch()

const loader = new CheerioWebBaseLoader("https://github.com/AyushRay06")

const docs = await loader.load()

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 20,
})
const splittedDocs = await splitter.splitDocuments(docs)

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
})

const vectorstore = await MemoryVectorStore.fromDocuments(
  splittedDocs,
  embeddings
)
const retriever = vectorstore.asRetriever({ k: 25 })

const retrieverTool = new createRetrieverTool(retriever, {
  name: "github-search",
  description:
    "Use this tool when user ask anything related to his github account",
})
const tools = [searchTool, retrieverTool]
// Chat History
const chatHistory = []

// Agent Creation
const agent = new createToolCallingAgent({
  llm,
  prompt,
  tools,
})

// To invoke agents we need AgentExecutor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
})

//just to get input from the terminal
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
})

function conversation() {
  rl.question("User: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close()
      return
    }
    const response = await agentExecutor.invoke({
      input: input,
      chat_history: chatHistory,
    })
    console.log("Agent: ", response.output)
    chatHistory.push(new HumanMessage(input))
    chatHistory.push(new AIMessage(response.output))

    conversation()
  })
}

conversation()
