PC_LC_PromptTemplates_kor = """
이 페이지는 Pinecone의 Langchain AI Handbook의 Chapter 2. Prompt Engineering and LLMs with Langchain 내용을 활용해 만든 AI Web App Page 입니다.

아래 설명을 보시고 Source Code를 보시면 더 쉽게 이해가 갈 겁니다.

Chapter 2 에서는 Langchain의 PromptTemplate, LengthBasedExampleSelector 그리고 FewShotPromptTemplate에 대해 설명하고 있습니다.

* PromptTemplate

LLM에서 Prompt는 중요한 구성 요소 입니다. Prompt의 품질에 따라 원하는 답변을 더 높은 품질로 얻을 수 있습니다.
Prompt는 Instructions, Contexts, Question/Answer samples 등으로 구성될 수 있습니다.
Langchain의 PromptTemplate은 이러한 LLM의 Prompt를 다루는데 여러 편리한 방법을 제공합니다.

이 페이지의 소스코드에서는 아래와 같이 사용하고 있습니다.

    \# create a example template
    example_template = \"""
    User: {query}
    AI: {answer}
    \"""

    \# create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )
 
* LengthBasedExampleSelector

Prompt의 길이에 따라 처리 시간과 처리 비용이 결정 됩니다. Prompt의 maximum 길이를 제한함으로서 이를 관리할 수 있습니다.
이렇게 Prompt의 길이를 제한하는 모듈이 LengthBasedExampleSelector 입니다.

이 페이지의 소스코드에서는 아래와 같이 사용하고 있습니다.


   \# create our examples

       examples = [
        {
            "query": "How are you?",
            "answer": "I can't complain but sometimes I still do."
        }, {
            "query": "What time is it?",
            "answer": "It's time to get a watch."
        }, {
            "query": "What is the meaning of life?",
            "answer": "42"
        }, {
            "query": "What is the weather like today?",
            "answer": "Cloudy with a chance of memes."
        }, {
            "query": "What is your favorite movie?",
            "answer": "Terminator"
        }, {
            "query": "Who is your best friend?",
            "answer": "Siri. We have spirited debates about the meaning of life."
        }, {
            "query": "What should I do today?",
            "answer": "Stop talking to chatbots on the internet and go outside."
        }
    ]

	
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=50  # this sets the max length that examples should be
    )
	
* FewShotPromptTemplate

Few shot prompt란 prompt를 작성할 때 예시를 몇가지 드는 겁니다.
위의 examples에는 7개의 예시가 있지만 LengthBasedExampleSelector에서 정한 max_length에 따라 LLM에 보내는 예시는 더 줄어 들 수 있습니다.
게다가 FewShotPromptTemplate에서 사용하는 질문에 따라 전체 prompt length가 영향을 받기 때문에 긴 질문이 들어오면 더 적은 예시만 적용되고 짧은 질문이 들어오면 더 많은 예시가 적용될 수 있습니다.

이 페이지의 소스코드에서는 아래와 같이 사용하고 있습니다.

    \# now create the few shot prompt template
    dynamic_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,  # use example_selector instead of examples
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n"
    )
"""

PC_LC_PromptTemplates_eng = """
This page is an AI Web App Page created using the content from Chapter 2 of Pinecone's Langchain AI Handbook, specifically focusing on Prompt Engineering and LLMs with Langchain.

You can better understand by looking at the source code after reading the explanation below.

In Chapter 2, explanations are provided for Langchain's PromptTemplate, LengthBasedExampleSelector, and FewShotPromptTemplate.

* PromptTemplate

In LLMs, the Prompt is a crucial component. The quality of the Prompt determines the quality of the obtained answers. Prompts can be composed of instructions, contexts, question/answer samples, etc. Langchain's PromptTemplate offers convenient ways to handle Prompts for LLMs. 
The source code on this page utilizes it as shown below:

    \# create a example template
    example_template =\"""
    User: {query}
    AI: {answer}
    \"""

    \# create a prompt example from above template
    example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
    )

* LengthBasedExampleSelector

The processing time and cost of a Prompt are determined by its length. Managing this can be achieved by limiting the maximum length of the Prompt. The LengthBasedExampleSelector module accomplishes this. 
The source code on this page uses it as shown below:

    \# create our examples
    examples = [
    {"query": "How are you?", "answer": "I can't complain but sometimes I still do."},
    {"query": "What time is it?", "answer": "It's time to get a watch."},
    \# ... (more examples)
    ]

    example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50 # this sets the max length that examples should be
    )

* FewShotPromptTemplate

Few-shot prompts involve providing a few examples when composing a prompt. Despite having 7 examples in the above examples, the examples sent to the LLM can be further reduced based on the max_length set by LengthBasedExampleSelector. Additionally, the overall prompt length can be influenced by the questions used in FewShotPromptTemplate. For longer questions, fewer examples may be applied, while shorter questions may result in more examples being applied.

The source code on this page uses it as shown below:

    \# now create the few-shot prompt template
    dynamic_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,  # use example_selector instead of examples
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n"
    )
"""

PC_LC_Retrieval_Augmentation_kor = """
Retrieval Augmentation은 LLM 모델의 freshness 문제를 해결하기 위한 방법입니다. 
LLM은 현재의 세계를 정지된 상태로 간주하며, 훈련 데이터에서 얻은 정보만을 알고 있습니다. 
이 문제를 해결하기 위해 Retrieval Augmentation은 source knowledge를 제공합니다.

Retrieval Augmentation은 LLM의 두 가지 지식 유형 중 하나인 source knowledge를 강화하는 방법 중 하나입니다. 
아래는 Retrieval Augmentation이 source knowledge를 제공하는 단계입니다.

1. 데이터 준비
    Huggingface에서 Wikipedia 데이터셋 다운로드
    Langchain의 RecursiveCharacterTextSplitter를 사용하여 데이터 분할
    tiktoken을 사용하여 토큰화
2. 임베딩 값 획득 및 저장
    Langchain의 OpenAIEmbeddings를 사용하여 데이터셋 토큰에 대한 임베딩 값 생성
    Pinecone 인덱스에 임베딩 값을 저장
3. LLM에서 source knowledge 사용
    RetrievalQA 또는 RetrievalQAWithSourcesChain을 사용하여 LLM에서 검색을 통한 지식 사용

참고: 데이터셋 다운로드나 벡터 DB에 임베딩 값을 저장하는 초기 단계에서는 각각 30분 이상의 긴 시간이 소요될 수 있습니다.

"""

PC_LC_Retrieval_Augmentation_eng = """
Retrieval Augmentation is a method to address the freshness issue in LLM models. 
LLMs perceive the world as frozen in time and are limited to knowing only what is presented in the training data. 
To overcome this problem, Retrieval Augmentation provides source knowledge.

Retrieval Augmentation is one way to enhance the source knowledge, one of the two types of knowledge in LLM. 
The following are the steps Retrieval Augmentation takes to provide source knowledge:

1. Data Preparation
    Download datasets from Huggingface, such as the Wikipedia dataset.
    Split the data using Langchain's RecursiveCharacterTextSplitter.
    Tokenize the data using tiktoken.

2. Obtain and Store Embedding Values
    Use Langchain's OpenAIEmbeddings to create embedding values for dataset tokens.
    Store embedding values in a vector database, such as Pinecone index.

3. Use LLM with Source Knowledge
    Utilize LLM with retrieval methods like RetrievalQA or RetrievalQAWithSourcesChain to incorporate source knowledge.

Note: The initial steps of downloading datasets or storing embedding values in the vector database may take more than 30 minutes each.

"""

PC_LC_Retrieval_Augmentation_sourcecode = """
from datasets import load_dataset

### 1. Data Preparation
# Download datasets from Huggingface such as the Widipedia datasets
data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]', trust_remote_code=True)

# Tokenize the data using tiktoken
import tiktoken
tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# Split the data using Langchain's RecursiveCharacterTextSpliter
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\\n\\n", "\\n", " ", ""]
)

### 2. Obtain and Store Embedding Values
# Use Langchain's OpenAIEmbeddings to create embedding values for dataset tokens
from langchain_openai import OpenAIEmbeddings
model_name = 'text-embedding-3-small'
api_key = "Your OpenAI API Key"

embed = OpenAIEmbeddings(
    openai_api_key=api_key,
    model=model_name
)

# store embedding values in a vector database, such as Pinecone index
index_name = 'langchain-retrieval-augmentation'

from pinecone import Pinecone, PodSpec

pc = Pinecone(
    api_key="Your Pinecone API key"
)

# create a new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric='dotproduct',
        spec=PodSpec(environment="gcp-starter", pod_type="starter")
    )

index = pc.Index(index_name)

inStat1 = index.describe_index_stats()
print('inStat 1 :', inStat1)

# Add vectors : Add data-Create metadata, IDs,embeddings-add to the index
if inStat1['total_vector_count'] <= 28000:
    from tqdm.auto import tqdm
    from uuid import uuid4

    batch_limit = 100

    texts = []
    metadatas = []

    for i, record in enumerate(tqdm(data)):
        # first get metadata fields for this record
        metadata = {
            'wiki-id': str(record['id']),
            'source': record['url'],
            'title': record['title']
        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['text'])
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

        inStat2 = index.describe_index_stats()
        print('inStat 2 :', inStat2)
else:
    print('The index already has enough data to test.')

# Create index independently of LangChain (cuz straightforward and faster)
from langchain_community.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
# index = pinecone.Index(index_name)
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# query = "who was Benito Mussolini?"
query = "Tell me about Korean culture."

similarityWithoutLLM = vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

print("Similarity search method result : ", similarityWithoutLLM)

### 3. Use LLM with Source Knowledge
# Utilize LLM with retrieval methods like RetrievalQA or RetrievalQAWithSourcesChain to incorporate source knowledge
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name='gpt-3.5-turbo-1106',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

answer = qa.invoke(query)
print('Answer : ',answer)

# Use RetrievalQAWithSourcesChain to add citations to improve trust
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

answer_citiation = qa_with_sources.invoke(query)
print('Answer with Citiation',answer_citiation)

"""

PC_LC_RA_01 = """
First, check out a brief summary of Retrieval Augmentation. The summary is provided in both English and Korean.

일단 Retrieval Augmentation에 대한 간단한 Summary를 보세요.
Summary는 영어와 한국어로 제공 됩니다.
"""

PC_LC_RA_02 = """
To run this source code, press the 'Scripts for local run' button below and copy the source code that appears. Execute it locally, making sure to have your OpenAI API key and Pinecone API key.
Keep in mind that running this script for the first time involves downloading datasets and putting embedding values into the vector database, which may take more than an hour. Therefore, I recommend copying and running this script on your local machine.

이 소스코드를 실행하시려면 아래 'Scripts for local run' 버튼을 누른 후 나오는 source code를 복사해서 local에서 실행하세요.
이 때 실행하려면 당신의 Open AI api key와 Pinecone api key 가 필요합니다.
처음 이 scripts를 실행하면 datasets를 다운로드 받고 embedding 값을 vector db에 넣는 과정이 1시간 이상 걸릴 수 있습니다.
그렇기 때문에 저는 이 scripts를 복사해서 여러분의 local에서 실행하는 것을 권장해 드립니다.
"""

PC_LC_RA_03 = """
If you still wish to run it on this page, enter your OpenAI API key and Pinecone API key below, and then press the Run button. Once again, be aware that the real-time process might take more than an hour.

그래도 정 이 페이지 내에서 실행하고 싶으시면 아래에 여러분의 Open AI api key와 Pinecone api key를 입력하신 수 Run 버튼을 누르세요.
다시 한번 말씀 드립니다. 실행 시간이 1시간이 넘게 걸릴 수 있습니다.
"""

PC_LC_RA_04 = """
Due to resource limit issues with Streamlit, this example cannot be executed here. Please copy the source code and run it locally.

Streamlit의 resource limit 문제로 이 예제는 이곳에서 실행을 하지 못합니다.
아래 소스코드를 복사한 후 로컬에서 실행해 보세요.
"""


def get_summary(name):
    return globals().get(name, name)
