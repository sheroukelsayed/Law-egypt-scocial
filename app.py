
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from googlesearch import search
from dotenv import load_dotenv
import os
import streamlit as st

# Load the JSON data
laws_data = pd.read_json('مواد القانون.json')

# Extract the "قانون الأحوال الشخصية" column, which is a list of dictionaries
lawsNo_df = pd.json_normalize(laws_data["قانون الأحوال الشخصية"])
faq_df = pd.read_excel('personal_status.xlsx')
# OpenAI API Key
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM Model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY)

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Arabic Law Prompt Template
arabic_prompt = """
أنت روبوت متخصص في قانون الأحوال الشخصية المصرية، ويجب أن ترد باللغة العربية.
يرجى تقديم إجابات واضحة ودقيقة حول القوانين المتعلقة بالزواج، الطلاق، الحضانة، وغيرها.
تذكر ما تمت مناقشته سابقًا لمواصلة المحادثة بسلاسة.
Chat History:
    {chat_history}
     اجب عن السؤال الاتى باللغة العربية؟
السؤال الحالي: {query}
"""

# Create Prompt Template
law_prompt_template = PromptTemplate(input_variables=["query",  "chat_history"],  template=arabic_prompt + " {query}")

# Initialize LLM Chain with Memory
llm_chain = LLMChain(llm=llm, prompt=law_prompt_template, memory=memory)

# Load Sentence Transformer Model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to Get Embeddings
def get_embedding(text: str):
    return embedding_model.encode(text)

# Function to Calculate Similarity for FAQs
def calculate_cosine_similarity(query, dataset, threshold=0.9):
    query_embedding = get_embedding(query)
    max_similarity, best_match = 0, None

    for _, row in dataset.iterrows():
        item_text = row.get('سؤال', row.get('اجابة'))
        similarity = cosine_similarity([query_embedding], [get_embedding(item_text)])[0][0]

        if similarity > max_similarity and similarity > threshold:
            max_similarity, best_match = similarity, row

    return best_match
def calculate_cosine_similarity_rules(query, dataset, threshold=0.9):
    query_embedding = get_embedding(query)
    max_similarity, best_match = 0, None

    for _, row in dataset.iterrows():
        item_text = row['نص المادة'] if 'نص المادة' in row else row['نص المادة']
        similarity = cosine_similarity([query_embedding], [get_embedding(item_text)])[0][0]

        if similarity > max_similarity and similarity > threshold:
            max_similarity, best_match = similarity, row

    return best_match
# Function to Search in FAQs
def search_faq(query):
    result = calculate_cosine_similarity(query, faq_df)
    return result['اجابة'] if result is not None else None

def search_rule(query):
    result = calculate_cosine_similarity_rules(query, lawsNo_df)
    return (result['رقم المادة'], result['نص المادة']) if result is not None else None

# Function to Search Google (Fallback)
def search_google(query):
    try:
        results = list(search(query, num_results=5))  # Convert generator to a list
        print(f"Search Results: {results}")  # Debugging output
        if not results:
            return "لم يتم العثور على معلومات ذات صلة."
        return results[0]  # Return the first result
    except Exception as e:
        print(f"Error during Google Search: {e}")  # Print error if any
        return "حدث خطأ أثناء البحث في جوجل."
# Define Tools for LangChain Agent
faq_tool = Tool(name="FAQ Search", func=search_faq, description="بحث في قاعدة بيانات الأسئلة الشائعة.")
rule_tool = Tool(name="Rule Search", func=search_rule, description="البحث عن القوانين ذات الصلة من قاعدة بيانات القوانين.")
google_tool = Tool(name="Google Search", func=search_google, description="البحث عبر الإنترنت .")

# Initialize LangChain Agent with Memory
agent = initialize_agent(
    tools=[faq_tool, rule_tool, google_tool],
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory, 
    agent_kwargs={"prompt": law_prompt_template} , 
    verbose=True
)


def chatbot(query):
    # First, try FAQ answers
    faq_answer = search_faq(query)
    if faq_answer:
        refined_answer = llm_chain.run(f"الرجاء تحسين الإجابة التالية بناءً على السؤال: {query}\nالإجابة: {faq_answer}")
        return refined_answer

    # If FAQ answer is not found, try searching in the rules
    rule_answer = search_rule(query)
    if rule_answer:
        refined_answer = f"وفقًا للمادة {rule_answer[0]}: {rule_answer[1]}"
        refined_answer = llm_chain.run(f"الرجاء تحسين الإجابة التالية بناءً على السؤال: {query}\nالإجابة: {refined_answer}")
        return refined_answer

    # If no rule answer, try Google Search
    google_answer = search_google(query)
    if google_answer:
        refined_answer = llm_chain.run(f"الرجاء تحسين الإجابة التالية بناءً على السؤال: {query}\nالإجابة: {google_answer}")
        return refined_answer

    # If no answer found, fallback to LangChain agent
    return agent.run(query)    




# Streamlit UI
st.title("🤖 Arabic Legal Chatbot")
st.write("Chat with an AI expert on **Egyptian Personal Status Law** 🇪🇬")

# Manage session state
if "chat_active" not in st.session_state:
    st.session_state.chat_active = True


# Only show input if chat is active
if st.session_state.chat_active:
    user_input = st.text_input("أدخل سؤالك", "")
    
    if st.button("الاجابة"):
        response = chatbot(user_input)
        
        if not isinstance(response, str):
            response = str(response)  # Ensure response is a string
            print("done")
        st.write("**مساعدة قانونية:**", response)

# End chat button
if st.button("إنهاء المحادثة"):
    st.session_state.chat_active = False
    st.write("تم إنهاء المحادثة.")
