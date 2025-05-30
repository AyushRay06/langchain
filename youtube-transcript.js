import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"
import * as dotenv from "dotenv"
import { YoutubeLoader } from "@langchain/community/document_loaders/web/youtube"
dotenv.config()

//LLM
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-pro",
  temperature: 0.5,
  maxRetries: 3,
})

//PROMPT
const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the user Question:{input} based on the Context:{context}`
)

const loader = YoutubeLoader.createFromUrl(
  "https://www.youtube.com/watch?v=ROrtQLKvj4w",
  {
    language: "en",
  }
)
const docs = await loader.load()

const chain = await createStuffDocumentsChain({
  llm,
  prompt,
})

const response = await chain.invoke({
  input: "what all exercies he talked about",
  context: docs,
})

console.log(response)
