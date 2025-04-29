import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai"
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { TaskType } from "@google/generative-ai"
import * as dotenv from "dotenv"
import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { createRetrievalChain } from "langchain/chains/retrieval"
dotenv.config()

//LLM
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-pro",
  temperature: 0.5,
  maxRetries: 3,
})

//PROMPT
const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the users question from the following Context:{context} 
  Question:{input}`
)

//SCRAPER(Scrapes data from web-page fro context )
const loader = new CheerioWebBaseLoader(
  "https://portfolio-website-nu-lyart.vercel.app/"
)
const docs = await loader.load()

// SPLITTER(optimazation of the token size as more teh token size mare are we charged)
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 20,
})
const splittedDocs = await splitter.splitDocuments(docs)

// As now we have splitted the entier document into smaller documents so for answering
// the users question we need to peek the most relevent document
// so for that we use VECTOR DB and EMBEDINGS

// EMBEDINGS

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
})

// VECTOR STORE
const vectorstore = await MemoryVectorStore.fromDocuments(
  splittedDocs,
  embeddings
)

// RETRIEVER
const retriever = vectorstore.asRetriever({ k: 25 })

// CHAIN(flow of the proceses)
const chain = await createStuffDocumentsChain({
  llm,
  prompt,
})
// RETRIVER CHAIN
const retriverChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
})

//OUTPUT(here we pass in the input and invoke the chain)
const response = await retriverChain.invoke({
  input: "List some of his projects",
})

// console.log(docs[0])

console.log(response.answer)
