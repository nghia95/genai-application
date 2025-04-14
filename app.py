# pylint: disable=broad-exception-caught,broad-exception-raised,invalid-name
"""
This module demonstrates the usage of the Gemini API in Vertex AI within a Streamlit application.
"""
from dotenv import load_dotenv

load_dotenv()

import os

from google import genai
import google.auth
from google.genai.types import GenerateContentConfig, Part
import httpx
import streamlit as st
import pandas as pd
import scipy
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from langchain.document_loaders import WikipediaLoader
from langchain_community.document_loaders import WikipediaLoader,PyPDFLoader, WebBaseLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


def _project_id() -> str:
    """Use the Google Auth helper (via the metadata service) to get the Google Cloud Project"""
    try:
        _, project = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError as e:
        raise Exception("Could not automatically determine credentials") from e
    if not project:
        raise Exception("Could not determine project from credentials.")
    return project


def _region() -> str:
    """Use the local metadata service to get the region"""
    try:
        resp = httpx.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/region",
            headers={"Metadata-Flavor": "Google"},
        )
        return resp.text.split("/")[-1]
    except Exception:
        return "us-central1"


API_KEY = os.environ.get("GOOGLE_API_KEY")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", _project_id())
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", _region())
MODELS = {
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash-Lite",
    "gemini-2.0-flash-thinking-exp-01-21": "Gemini 2.0 Flash Thinking",
    "gemini-2.0-pro-exp-02-05": "Gemini 2.0 Pro",
}
EMBEDDING_MODEL = "text-embedding-004"
import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "My_project"

@st.cache_resource
def load_client() -> genai.Client:
    """Load Google Gen AI Client."""
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        api_key=API_KEY,
    )


def get_model_name(name: str | None) -> str:
    """Get the formatted model name."""
    if not name:
        return "Gemini"
    return MODELS.get(name, "Gemini")


st.link_button(
    "View on GitHub",
    "https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/sample-apps/gemini-streamlit-cloudrun",
)

cloud_run_service = os.environ.get("K_SERVICE")
if cloud_run_service:
    st.link_button(
        "Open in Cloud Run",
        f"https://console.cloud.google.com/run/detail/us-central1/{cloud_run_service}/source",
    )

st.header(":sparkles: Gemini API in Vertex AI", divider="rainbow")
client = load_client()

freeform_tab, wiki_tab, pdf_tab, tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Freeform",
        "Wikipedia QA",
        "PDF QA",
        "Generate story",
        "Marketing campaign",
        "Image Playground",
        "Video Playground",
    ]
)


with freeform_tab:
    st.subheader("Enter Your Own Prompt")

    selected_model = st.radio(
        "Select Model:",
        MODELS.keys(),
        format_func=get_model_name,
        key="selected_model_freeform",
        horizontal=True,
    )

    temperature = st.slider(
        "Select the temperature (Model Randomness):",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.05,
        key="temperature",
    )

    max_output_tokens = st.slider(
        "Maximum Number of Tokens to Generate:",
        min_value=1,
        max_value=8192,
        value=2048,
        step=1,
        key="max_output_tokens",
    )

    top_p = st.slider(
        "Select the Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        key="top_p",
    )

    prompt = st.text_area(
        "Enter your prompt here...",
        key="prompt",
        height=200,
    )

    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
    )

    generate_freeform = st.button("Generate", key="generate_freeform")
    if generate_freeform and prompt:
        with st.spinner(
            f"Generating response using {get_model_name(selected_model)} ..."
        ):
            first_tab1, first_tab2 = st.tabs(["Response", "Prompt"])
            with first_tab1:
                response = client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config,
                ).text

                if response:
                    st.markdown(response)
            with first_tab2:
                st.markdown(
                    f"""Parameters:\n- Model ID: `{selected_model}`\n- Temperature: `{temperature}`\n- Top P: `{top_p}`\n- Max Output Tokens: `{max_output_tokens}`\n"""
                )
                st.code(prompt, language="markdown")

with wiki_tab:
    st.subheader("Enter Your Own Prompt about something in Wikipedia")

    selected_model_wiki = st.radio(
        "Select Model:",
        MODELS.keys(),
        format_func=get_model_name,
        key="selected_model_wiki",
        horizontal=True,
    )

    topic = st.text_input(
        "Enter the topic to search on wiki: \n\n", key="wiki_topic", value="Large Language Model"
    )

    prompt_wiki = st.text_area(
        "Enter your prompt here...",
        key="prompt_wiki",
        height=200,
    )


    generate_form_wiki = st.button("Generate", key="generate_freeform_wiki")

    
    if generate_form_wiki and prompt_wiki:
        with st.spinner(
            f"Generating response using {get_model_name(selected_model_wiki)} ..."
        ):
            first_tab1_wiki, first_tab2_wiki = st.tabs(["Response", "Prompt"])
            docs = WikipediaLoader(query=topic, load_max_docs=10).load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=400,
                length_function=len,
            )

            chunks = text_splitter.split_documents(docs)
            embedding = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)

            # set persist directory so the vector store is saved to disk
            db = Chroma.from_documents(chunks, embedding, persist_directory="./vectorstore")

        # vector store
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10},  # number of nearest neighbors to retrieve
            )

        # You can also set temperature, top_p, top_k
            llm = VertexAI(model_name=selected_model_wiki, max_output_tokens=1024)

        # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        # rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # q/a chain
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            #response_wiki = rag_chain.invoke({"input":prompt_wiki})
            
            with first_tab1_wiki:
                response_wiki = rag_chain.invoke(prompt_wiki)
                citations = {doc.metadata["source"] for doc in response_wiki["source_documents"]}

               
                # response_wiki = client.models.generate_content(
                #     model=selected_model_wiki,
                #     contents=prompt_wiki,
                #     config=config_wiki,
                # ).text

                if response_wiki:
                    #st.markdown(response_wiki['answer'])
                    st.markdown(response_wiki["result"])
                    st.markdown(f"Citations: {citations}\n")
            with first_tab2_wiki:
                st.markdown(
                    f"""Parameters:\n- Model ID: `{selected_model_wiki}`\n"""
                )
                st.code(prompt_wiki, language="markdown")
            
            # with source:
            #     sources = response_wiki['context'][0]
            #     st.markdown(sources)

with pdf_tab:
    st.subheader("Enter Your Own Prompt about something in Wikipedia")

    selected_model_pdf = st.radio(
        "Select Model:",
        MODELS.keys(),
        format_func=get_model_name,
        key="selected_model_pdf",
        horizontal=True,
    )

    file_path = st.text_input(
        #"Enter the database path: \n\n", key="file_path", value="https://abc.xyz/assets/investor/static/pdf/20230203_alphabet_10K.pdf"
        "Enter the database path: \n\n", key="file_path", value="../data"
    )


    prompt_pdf = st.text_area(
        "Enter your prompt here...",
        key="prompt_pdf",
        height=200,
    )


    generate_form_pdf = st.button("Generate", key="generate_freeform_pdf")

    
    if generate_form_pdf and prompt_pdf:
        with st.spinner(
            f"Generating response using {get_model_name(selected_model_wiki)} ..."
        ):
            first_tab1_pdf, first_tab2_pdf = st.tabs(["Response", "Prompt"])

            documents = []
            for file in os.listdir(file_path):
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(file_path, file)
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
            #documents = PyPDFLoader(file_path).load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=40,
                length_function=len,
            )

            chunks_pdf = text_splitter.split_documents(documents)
            embedding_pdf = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)

            # set persist directory so the vector store is saved to disk
            db_pdf = Chroma.from_documents(chunks_pdf, embedding_pdf, persist_directory="./vectorstore")

        # vector store
            retriever_pdf = db_pdf.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10},  # number of nearest neighbors to retrieve
            )

        # You can also set temperature, top_p, top_k
            llm_pdf = VertexAI(model_name=selected_model_pdf, max_output_tokens=1024)

        # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        # rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # q/a chain
            rag_chain_pdf = RetrievalQA.from_chain_type(
                llm=llm_pdf,
                chain_type="stuff",
                retriever=retriever_pdf,
                return_source_documents=True,
            )
            #response_wiki = rag_chain.invoke({"input":prompt_wiki})
            
            with first_tab1_pdf:
                response_pdf = rag_chain_pdf.invoke(prompt_pdf)
                citations_pdf = {doc.metadata["source"] for doc in response_pdf["source_documents"]}

               
                # response_wiki = client.models.generate_content(
                #     model=selected_model_wiki,
                #     contents=prompt_wiki,
                #     config=config_wiki,
                # ).text

                if response_pdf:
                    #st.markdown(response_wiki['answer'])
                    st.markdown(response_pdf["result"])
                    st.markdown(f"Citations: {citations_pdf}\n")
            with first_tab2_pdf:
                st.markdown(
                    f"""Parameters:\n- Model ID: `{selected_model_pdf}`\n"""
                )
                st.code(prompt_pdf, language="markdown")

with tab1:
    st.subheader("Generate a story")

    selected_model = st.radio(
        "Select Model:",
        MODELS.keys(),
        format_func=get_model_name,
        key="selected_model_story",
        horizontal=True,
    )

    # Story premise
    character_name = st.text_input(
        "Enter character name: \n\n", key="character_name", value="Mittens"
    )
    character_type = st.text_input(
        "What type of character is it? \n\n", key="character_type", value="Cat"
    )
    character_persona = st.text_input(
        "What personality does the character have? \n\n",
        key="character_persona",
        value="Mittens is a very friendly cat.",
    )
    character_location = st.text_input(
        "Where does the character live? \n\n",
        key="character_location",
        value="Andromeda Galaxy",
    )
    story_premise = st.multiselect(
        "What is the story premise? (can select multiple) \n\n",
        [
            "Love",
            "Adventure",
            "Mystery",
            "Horror",
            "Comedy",
            "Sci-Fi",
            "Fantasy",
            "Thriller",
        ],
        key="story_premise",
        default=["Love", "Adventure"],
    )
    creative_control = st.radio(
        "Select the creativity level: \n\n",
        ["Low", "High"],
        key="creative_control",
        horizontal=True,
    )
    length_of_story = st.radio(
        "Select the length of the story: \n\n",
        ["Short", "Long"],
        key="length_of_story",
        horizontal=True,
    )

    if creative_control == "Low":
        temperature = 0.30
    else:
        temperature = 0.95

    if length_of_story == "Short":
        max_output_tokens = 2048
    else:
        max_output_tokens = 8192

    prompt = f"""Write a {length_of_story} story based on the following premise: \n
  character_name: {character_name} \n
  character_type: {character_type} \n
  character_persona: {character_persona} \n
  character_location: {character_location} \n
  story_premise: {",".join(story_premise)} \n
  If the story is "short", then make sure to have 5 chapters or else if it is "long" then 10 chapters.
  Important point is that each chapters should be generated based on the premise given above.
  First start by giving the book introduction, chapter introductions and then each chapter. It should also have a proper ending.
  The book should have prologue and epilogue.
  """
    config = GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_output_tokens
    )

    generate_t2t = st.button("Generate my story", key="generate_t2t")
    if generate_t2t and prompt:
        with st.spinner(
            f"Generating your story using {get_model_name(selected_model)} ..."
        ):
            first_tab1, first_tab2 = st.tabs(["Story", "Prompt"])
            with first_tab1:
                response = client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config,
                ).text

                if response:
                    st.write("Your story:")
                    st.write(response)
            with first_tab2:
                st.markdown(
                    f"""Parameters:\n- Model ID: `{selected_model}`\n- Temperature: `{temperature}`\n- Max Output Tokens: `{max_output_tokens}`\n"""
                )
                st.code(prompt, language="markdown")

with tab2:
    st.subheader("Generate your marketing campaign")

    selected_model = st.radio(
        "Select Model:",
        MODELS,
        format_func=get_model_name,
        key="selected_model_marketing",
        horizontal=True,
    )

    product_name = st.text_input(
        "What is the name of the product? \n\n", key="product_name", value="ZomZoo"
    )
    product_category = st.radio(
        "Select your product category: \n\n",
        ["Clothing", "Electronics", "Food", "Health & Beauty", "Home & Garden"],
        key="product_category",
        horizontal=True,
    )
    st.write("Select your target audience: ")
    target_audience_age = st.radio(
        "Target age: \n\n",
        ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        key="target_audience_age",
        horizontal=True,
    )
    target_audience_location = st.radio(
        "Target location: \n\n",
        ["Urban", "Suburban", "Rural"],
        key="target_audience_location",
        horizontal=True,
    )
    st.write("Select your marketing campaign goal: ")
    campaign_goal = st.multiselect(
        "Select your marketing campaign goal: \n\n",
        [
            "Increase brand awareness",
            "Generate leads",
            "Drive sales",
            "Improve brand sentiment",
        ],
        key="campaign_goal",
        default=["Increase brand awareness", "Generate leads"],
    )
    if campaign_goal is None:
        campaign_goal = ["Increase brand awareness", "Generate leads"]
    brand_voice = st.radio(
        "Select your brand voice: \n\n",
        ["Formal", "Informal", "Serious", "Humorous"],
        key="brand_voice",
        horizontal=True,
    )
    estimated_budget = st.radio(
        "Select your estimated budget ($): \n\n",
        ["1,000-5,000", "5,000-10,000", "10,000-20,000", "20,000+"],
        key="estimated_budget",
        horizontal=True,
    )

    prompt = f"""Generate a marketing campaign for {product_name}, a {product_category} designed for the age group: {target_audience_age}.
  The target location is this: {target_audience_location}.
  Aim to primarily achieve {campaign_goal}.
  Emphasize the product's unique selling proposition while using a {brand_voice} tone of voice.
  Allocate the total budget of {estimated_budget}.
  With these inputs, make sure to follow following guidelines and generate the marketing campaign with proper headlines: \n
  - Briefly describe company, its values, mission, and target audience.
  - Highlight any relevant brand guidelines or messaging frameworks.
  - Provide a concise overview of the campaign's objectives and goals.
  - Briefly explain the product or service being promoted.
  - Define your ideal customer with clear demographics, psychographics, and behavioral insights.
  - Understand their needs, wants, motivations, and pain points.
  - Clearly articulate the desired outcomes for the campaign.
  - Use SMART goals (Specific, Measurable, Achievable, Relevant, and Time-bound) for clarity.
  - Define key performance indicators (KPIs) to track progress and success.
  - Specify the primary and secondary goals of the campaign.
  - Examples include brand awareness, lead generation, sales growth, or website traffic.
  - Clearly define what differentiates your product or service from competitors.
  - Emphasize the value proposition and unique benefits offered to the target audience.
  - Define the desired tone and personality of the campaign messaging.
  - Identify the specific channels you will use to reach your target audience.
  - Clearly state the desired action you want the audience to take.
  - Make it specific, compelling, and easy to understand.
  - Identify and analyze your key competitors in the market.
  - Understand their strengths and weaknesses, target audience, and marketing strategies.
  - Develop a differentiation strategy to stand out from the competition.
  - Define how you will track the success of the campaign.
  - Utilize relevant KPIs to measure performance and return on investment (ROI).
  Give proper bullet points and headlines for the marketing campaign. Do not produce any empty lines.
  Be very succinct and to the point.
  """

    config = GenerateContentConfig(temperature=0.8, max_output_tokens=8192)

    generate_t2t = st.button("Generate my campaign", key="generate_campaign")
    if generate_t2t and prompt:
        second_tab1, second_tab2 = st.tabs(["Campaign", "Prompt"])
        with st.spinner(
            f"Generating your marketing campaign using {get_model_name(selected_model)} ..."
        ):
            with second_tab1:
                response = client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config,
                ).text
                if response:
                    st.write("Your marketing campaign:")
                    st.write(response)
            with second_tab2:
                st.code(prompt, language="markdown")

with tab3:
    st.subheader("Image Playground")

    selected_model = st.radio(
        "Select Model:",
        MODELS,
        format_func=get_model_name,
        key="selected_model_image",
        horizontal=True,
    )

    furniture, oven, er_diagrams, glasses, math_reasoning = st.tabs(
        [
            "Furniture recommendation",
            "Oven instructions",
            "ER diagrams",
            "Glasses recommendation",
            "Math reasoning",
        ]
    )

    with furniture:
        st.markdown(
            """In this demo, you will be presented with a scene (e.g., a living room) and will use the Gemini model to perform visual understanding. You will see how Gemini can be used to recommend an item (e.g., a chair) from a list of furniture options as input. You can use Gemini to recommend a chair that would complement the given scene and will be provided with its rationale for such selections from the provided list."""
        )

        room_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/rooms/living_room.jpeg"
        chair_1_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/chair1.jpeg"
        chair_2_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/chair2.jpeg"
        chair_3_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/chair3.jpeg"
        chair_4_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/chair4.jpeg"

        st.image(room_image_uri, width=350, caption="Image of a living room")
        st.image(
            [
                chair_1_image_uri,
                chair_2_image_uri,
                chair_3_image_uri,
                chair_4_image_uri,
            ],
            width=200,
            caption=["Chair 1", "Chair 2", "Chair 3", "Chair 4"],
        )

        st.write(
            "Our expectation: Recommend a chair that would complement the given image of a living room."
        )
        content = [
            "Consider the following chairs:",
            "chair 1:",
            Part.from_uri(file_uri=chair_1_image_uri, mime_type="image/jpeg"),
            "chair 2:",
            Part.from_uri(file_uri=chair_2_image_uri, mime_type="image/jpeg"),
            "chair 3:",
            Part.from_uri(file_uri=chair_3_image_uri, mime_type="image/jpeg"),
            "and",
            "chair 4:",
            Part.from_uri(file_uri=chair_4_image_uri, mime_type="image/jpeg"),
            "\n"
            "For each chair, explain why it would be suitable or not suitable for the following room:",
            Part.from_uri(file_uri=room_image_uri, mime_type="image/jpeg"),
            "Only recommend for the room provided and not other rooms. Provide your recommendation in a table format with chair name and reason as columns.",
        ]

        tab1, tab2 = st.tabs(["Response", "Prompt"])
        generate_image_description = st.button(
            "Generate recommendation....", key="generate_image_description"
        )
        with tab1:
            if generate_image_description and content:
                with st.spinner(
                    f"Generating recommendation using {get_model_name(selected_model)} ..."
                ):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=content,
                        config=config,
                    ).text
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.code(content, language="markdown")

    with oven:
        stove_screen_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/stove.jpg"
        st.write(
            "Equipped with the ability to extract information from visual elements on screens, Gemini can analyze screenshots, icons, and layouts to provide a holistic understanding of the depicted scene."
        )
        st.image(stove_screen_uri, width=350, caption="Image of a oven")
        st.write(
            "Our expectation: Provide instructions for resetting the clock on this appliance in English"
        )
        prompt = """How can I reset the clock on this appliance? Provide the instructions in English.
If instructions include buttons, also explain where those buttons are physically located.
"""
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        generate_instructions_description = st.button(
            "Generate instructions", key="generate_instructions_description"
        )
        with tab1:
            if generate_instructions_description and prompt:
                with st.spinner(
                    f"Generating instructions using {get_model_name(selected_model)}..."
                ):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[
                            Part.from_uri(
                                file_uri=stove_screen_uri, mime_type="image/jpeg"
                            ),
                            prompt,
                        ],
                    ).text
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.code(prompt, language="markdown")

    with er_diagrams:
        er_diag_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/er.png"
        st.write(
            "Gemini multimodal capabilities empower it to comprehend diagrams and take actionable steps, such as optimization or code generation. The following example demonstrates how Gemini can decipher an Entity Relationship (ER) diagram."
        )
        st.image(er_diag_uri, width=350, caption="Image of an ER diagram")
        st.write(
            "Our expectation: Document the entities and relationships in this ER diagram."
        )
        prompt = """Document the entities and relationships in this ER diagram.
        """
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        er_diag_img_description = st.button("Generate!", key="er_diag_img_description")
        with tab1:
            if er_diag_img_description and prompt:
                with st.spinner("Generating..."):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[
                            Part.from_uri(file_uri=er_diag_uri, mime_type="image/jpeg"),
                            prompt,
                        ],
                    ).text
        with tab2:
            st.write("Prompt used:")
            st.code(prompt, language="markdown")

    with glasses:
        compare_img_1_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/glasses1.jpg"
        compare_img_2_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/glasses2.jpg"

        st.write(
            """Gemini is capable of image comparison and providing recommendations. This can be useful in industries like e-commerce and retail.
          Below is an example of choosing which pair of glasses would be better suited to various face types:"""
        )
        face_type = st.radio(
            "What is your face shape?",
            ["Oval", "Round", "Square", "Heart", "Diamond"],
            key="face_type",
            horizontal=True,
        )
        output_type = st.radio(
            "Select the output type",
            ["text", "table", "json"],
            key="output_type",
            horizontal=True,
        )
        st.image(
            [compare_img_1_uri, compare_img_2_uri],
            width=350,
            caption=["Glasses type 1", "Glasses type 2"],
        )
        st.write(
            f"Our expectation: Suggest which glasses type is better for the {face_type} face shape"
        )
        content = [
            f"""Which of these glasses you recommend for me based on the shape of my face:{face_type}?
      I have an {face_type} shape face.
      Glasses 1: """,
            Part.from_uri(file_uri=compare_img_1_uri, mime_type="image/jpeg"),
            """
      Glasses 2: """,
            Part.from_uri(file_uri=compare_img_2_uri, mime_type="image/jpeg"),
            f"""
      Explain how you made to this decision.
      Provide your recommendation based on my face shape, and reasoning for each in {output_type} format.
      """,
        ]
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        compare_img_description = st.button(
            "Generate recommendation!", key="compare_img_description"
        )
        with tab1:
            if compare_img_description and content:
                with st.spinner(
                    f"Generating recommendations using {get_model_name(selected_model)}..."
                ):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[
                            Part.from_uri(file_uri=er_diag_uri, mime_type="image/jpeg"),
                            content,
                        ],
                    ).text
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.code(content, language="markdown")

    with math_reasoning:
        math_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/math_beauty.jpg"

        st.write(
            "Gemini can also recognize math formulas and equations and extract specific information from them. This capability is particularly useful for generating explanations for math problems, as shown below."
        )
        st.image(math_image_uri, width=350, caption="Image of a math equation")
        st.markdown(
            """
        Our expectation: Ask questions about the math equation as follows:
        - Extract the formula.
        - What is the symbol right before Pi? What does it mean?
        - Is this a famous formula? Does it have a name?
          """
        )
        prompt = """
Follow the instructions.
Surround math expressions with $.
Use a table with a row for each instruction and its result.

INSTRUCTIONS:
- Extract the formula.
- What is the symbol right before Pi? What does it mean?
- Is this a famous formula? Does it have a name?
"""
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        math_image_description = st.button(
            "Generate answers!", key="math_image_description"
        )
        with tab1:
            if math_image_description and prompt:
                with st.spinner(
                    f"Generating answers for formula using {get_model_name(selected_model)}..."
                ):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[
                            Part.from_uri(
                                file_uri=math_image_uri, mime_type="image/jpeg"
                            ),
                            prompt,
                        ],
                    ).text
                    st.markdown(response)
                    st.markdown("\n\n\n")
        with tab2:
            st.write("Prompt used:")
            st.code(prompt, language="markdown")

with tab4:
    st.subheader("Video Playground")

    selected_model = st.radio(
        "Select Model:",
        MODELS,
        format_func=get_model_name,
        key="selected_model_video",
        horizontal=True,
    )

    vide_desc, video_tags, video_highlights, video_geolocation = st.tabs(
        ["Video description", "Video tags", "Video highlights", "Video geolocation"]
    )

    with vide_desc:
        st.markdown(
            """Gemini can also provide the description of what is going on in the video:"""
        )
        video_desc_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/mediterraneansea.mp4"

        if video_desc_uri:
            st.video(video_desc_uri)
            st.write("Our expectation: Generate the description of the video")
            prompt = """Describe what is happening in the video and answer the following questions: \n
      - What am I looking at? \n
      - Where should I go to see it? \n
      - What are other top 5 places in the world that look like this?
      """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            vide_desc_description = st.button(
                "Generate video description", key="vide_desc_description"
            )
            with tab1:
                if vide_desc_description and prompt:
                    with st.spinner(
                        f"Generating video description using {get_model_name(selected_model)} ..."
                    ):
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                Part.from_uri(
                                    file_uri=video_desc_uri, mime_type="video/mp4"
                                ),
                                prompt,
                            ],
                        ).text
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.code(prompt, language="markdown")

    with video_tags:
        st.markdown(
            """Gemini can also extract tags throughout a video, as shown below:."""
        )
        video_tags_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/photography.mp4"

        if video_tags_uri:
            st.video(video_tags_uri)
            st.write("Our expectation: Generate the tags for the video")
            prompt = """Answer the following questions using the video only:
            1. What is in the video?
            2. What objects are in the video?
            3. What is the action in the video?
            4. Provide 5 best tags for this video?
            Give the answer in the table format with question and answer as columns.
      """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_tags_description = st.button(
                "Generate video tags", key="video_tags_description"
            )
            with tab1:
                if video_tags_description and prompt:
                    with st.spinner(
                        f"Generating video description using {get_model_name(selected_model)} ..."
                    ):
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                Part.from_uri(
                                    file_uri=video_tags_uri, mime_type="video/mp4"
                                ),
                                prompt,
                            ],
                        ).text
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.code(prompt, language="markdown")

    with video_highlights:
        st.markdown(
            """Below is another example of using Gemini to ask questions about objects, people or the context, as shown in the video about Pixel 8 below:"""
        )
        video_highlights_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4"

        if video_highlights_uri:
            st.video(video_highlights_uri)
            st.write("Our expectation: Generate the highlights for the video")
            prompt = """Answer the following questions using the video only:
What is the profession of the girl in this video?
Which all features of the phone are highlighted here?
Summarize the video in one paragraph.
Provide the answer in table format.
      """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_highlights_description = st.button(
                "Generate video highlights", key="video_highlights_description"
            )
            with tab1:
                if video_highlights_description and prompt:
                    with st.spinner(
                        f"Generating video highlights using {get_model_name(selected_model)} ..."
                    ):
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                Part.from_uri(
                                    file_uri=video_highlights_uri, mime_type="video/mp4"
                                ),
                                prompt,
                            ],
                        ).text
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.code(prompt, language="markdown")

    with video_geolocation:
        st.markdown(
            """Even in short, detail-packed videos, Gemini can identify the locations."""
        )
        video_geolocation_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/bus.mp4"

        if video_geolocation_uri:
            st.video(video_geolocation_uri)
            st.markdown(
                """Our expectation: \n
      Answer the following questions from the video:
        - What is this video about?
        - How do you know which city it is?
        - What street is this?
        - What is the nearest intersection?
      """
            )
            prompt = """Answer the following questions using the video only:
      What is this video about?
      How do you know which city it is?
      What street is this?
      What is the nearest intersection?
      Answer the following questions in a table format with question and answer as columns.
      """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_geolocation_description = st.button(
                "Generate", key="video_geolocation_description"
            )
            with tab1:
                if video_geolocation_description and prompt:
                    with st.spinner(
                        f"Generating location tags using {get_model_name(selected_model)} ..."
                    ):
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                Part.from_uri(
                                    file_uri=video_geolocation_uri,
                                    mime_type="video/mp4",
                                ),
                                prompt,
                            ],
                        ).text
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.code(prompt, language="markdown")