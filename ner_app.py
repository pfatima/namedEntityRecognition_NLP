import streamlit as st
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Download NLTK data
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function for rule-based NER using NLTK
def nltk_ner(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            entity_name = " ".join([token for token, pos in subtree.leaves()])
            entity_type = subtree.label()
            entities.append((entity_name, entity_type))
    return entities

# Function for NER using SpaCy's pre-trained model
def spacy_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit app
st.title("Named Entity Recognition (NER) App")
st.subheader("CTRL+ALT+DEFEAT")
st.write("""
This app demonstrates Named Entity Recognition (NER) using both rule-based (NLTK) and pre-trained (SpaCy) models.
""")


text = st.text_area("Enter text for NER:")

color_map = {
    "PERSON": "#ff311a",  # teal blue
    "ORG": "#FF8000",     # orchid purple
    "ORGANIZATION": "#FF8000",
    "GPE": "#628cf4"      # LimeGreen
}
def highlight_entities(text, entities):
    highlighted_text = text
    for entity, entity_type in entities:
        color = color_map.get(entity_type, "yellow")  # Default color if type not found
        highlighted_text = highlighted_text.replace(entity, f'<mark style="background-color: {color}; color: black;">{entity}</mark>')
    return highlighted_text

def style_entity(entity, entity_type):
    color = color_map.get(entity_type, "#000000")  # Default to black if type not found
    return f'<span style="color: {color};">{entity} ({entity_type})</span>'

if st.button("Analyze"):
    if text:
        st.subheader("Rule-based NER (NLTK)")
        nltk_entities = nltk_ner(text)
        if nltk_entities:
            for entity in nltk_entities:
                styled_entity = style_entity(entity[0], entity[1])
                st.markdown(styled_entity, unsafe_allow_html=True)
            # Highlight entities in the text
            highlighted_text = highlight_entities(text, nltk_entities)

            # Display the text with highlighted entities
            st.markdown(highlighted_text, unsafe_allow_html=True)
                
        else:
            st.write("No entities found.")

        st.subheader("Pre-trained NER (SpaCy)")
        spacy_entities = spacy_ner(text)
        if spacy_entities:
            for entity in spacy_entities:
                styled_entity = style_entity(entity[0], entity[1])
                st.markdown(styled_entity, unsafe_allow_html=True)
            # Highlight entities in the text
            highlighted_text = highlight_entities(text, spacy_entities)

            # Display the text with highlighted entities
            st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            st.write("No entities found.")
    
    else:
        st.write("Please enter text to analyze.")
        
