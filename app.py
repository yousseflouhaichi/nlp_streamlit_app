import streamlit as st
import torch
import re
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM


# Configure Streamlit page and state
st.set_page_config(page_title="LinguaLand", page_icon=":pencil2:")

# Remove anchor links
def hide_anchor_link():
    st.markdown("""
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """, unsafe_allow_html=True)

hide_anchor_link()

st.markdown("<h1 style='text-align: center; color: #31333F; font-size: 60px;'>LinguaLand</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: #31333F; font-size: 15px; font-style: italic'>Enhanced Text Generator ✒️</h6>", unsafe_allow_html=True)

st.markdown("</br>", unsafe_allow_html=True)
# Add background image
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-attachment: scroll; 
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
    
add_bg_from_local('assets/background/background_image.jpg')    

# Load the saved GPT-2 model
@st.cache_resource(show_spinner='Loading GPT-2 model, please wait...')
def load_GPT2_model(PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(PATH)
    model.eval()
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(PATH)
    return model, tokenizer, device

model, tokenizer, device = load_GPT2_model('models/finetuned_gpt2')


def load_trans_model(PATH):
    model = AutoModelForSeq2SeqLM.from_pretrained(PATH)
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    return model, tokenizer

# Load the saved traslation models
@st.cache_resource(show_spinner='Loading translation model, please wait...')
def load_trans_models():
    model_ar, tokenizer_ar = load_trans_model('models/trans_model_ar')
    model_fr, tokenizer_fr = load_trans_model('models/trans_model_fr')
    assert(model_ar != model_fr)
    return tokenizer_ar, model_ar, tokenizer_fr, model_fr

tokenizer_ar, model_ar, tokenizer_fr, model_fr = load_trans_models()


# Define the function to generate enhanced text
@st.cache_data(show_spinner='Generating, please wait...')
def generate(inp, output_type):

    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a , max_length=100)
    output = tokenizer.decode(output[0],skip_special_tokens=True)
    
    # Define regular expression patterns for each type of text
    pos_pattern = r"\[enhanced positively text\]:([^[\]]+)"
    neg_pattern = r"\[enhanced negatively text\]:([^[\]]+)"

    # Extract the positively enhanced and negatively enhanced text from the input string
    pos_text = re.findall(pos_pattern, output)[0].strip()
    neg_text = re.findall(neg_pattern, output)[0].strip()

    if output_type == "Positive":
        return pos_text
    elif output_type == "Negative":
        return neg_text
    
# Define the function to generate translation
@st.cache_data(show_spinner='Translating, please wait...')
def translate(text, lang):
    if(lang=='ar'):
        tr_tokenizer = tokenizer_ar
        tr_model = model_ar
    elif(lang=='fr'):
        tr_tokenizer = tokenizer_fr
        tr_model = model_fr
    input_ids = tr_tokenizer.encode(text, return_tensors="pt")
    outputs = tr_model.generate(input_ids, max_length=512)
    decoded = tr_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded


if "generate" not in st.session_state:
    st.session_state["generate"] = False
    
if "output" not in st.session_state:
    st.session_state["output"] = ""


# Define the Streamlit app
with st.form("my_form"):
    sentence = st.text_input("Enter a sentence :keyboard:")
    output_type = st.radio("Select output type", options=["Positive", "Negative"])
    submit_button = st.form_submit_button(label='Generate Enhanced Text', use_container_width=True, )


output_title_container = st.empty()
output_container = st.empty() 
translated_output_title = st.empty()  
translated_output = st.empty()

if(submit_button):
    # Check if sentence is empty
    if not sentence:
        st.error("Please enter a sentence")
        st.session_state["generate"] = False
    elif(st.session_state["generate"] == False):
        st.session_state["generate"] = True
        

if(st.session_state["generate"]):

    if(submit_button):
        # Generate enhanced text
        text = generate(sentence, output_type)
        st.session_state["output"] = text

    # Display enhanced text
    if output_type == "Positive":
        output_title_container.write("<h5 style='color: #004d99; padding-top: 10px; font-weight: bold;'>Positively Enhanced Sentence</h5>", unsafe_allow_html=True)
        output_container.info(st.session_state["output"])

    elif output_type == "Negative":
        output_title_container.write("<h5 style='color: #cc0000; padding-top: 10px; font-weight: bold;'>NEGATIVELY ENHANCED TEXT</h5>", unsafe_allow_html=True)
        output_container.error(st.session_state["output"])

    
    # Display a button with the icon
    col1, col2 = st.columns(2)
    with col1:
        #st.write("<a href='#' style='margin-right:10px;'><img src='https://cdn-icons-png.flaticon.com/512/197/197560.png' height='35'></a>", unsafe_allow_html=True)
        col1_1, col1_2 = st.columns([1,9])
        col1_1.write("<span style='margin-right:10px;'><img src='https://cdn-icons-png.flaticon.com/512/197/197578.png' height='35'></span>", unsafe_allow_html=True)
        if(col1_2.button("Translate to Arabic", use_container_width=True)):
            translated_output_title.write("<h5 style='color: #31333F; padding-top: 10px; font-weight: bold;'>Arabic Translation</h5>", unsafe_allow_html=True)
            translated_output.write(translate(st.session_state["output"], 'ar'))
        
    with col2:
        col2_1, col2_2 = st.columns([1,9])
        col2_1.write("<span style='margin-right:10px;'><img src='https://cdn-icons-png.flaticon.com/512/197/197560.png' height='35'></span>", unsafe_allow_html=True)
        if(col2_2.button("Translate to French", use_container_width=True)):
            translated_output_title.write("<h5 style='color: #31333F; padding-top: 10px; font-weight: bold;'>French Translation</h5>", unsafe_allow_html=True)
            translated_output.write(translate(st.session_state["output"], 'fr'))




# Hide menu and footer
#st.markdown("""
#<style>
#    #MainMenu, footer {visibility: hidden;}
#</style>
#""",unsafe_allow_html=True)