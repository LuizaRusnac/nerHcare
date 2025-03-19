import random
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from transformers import pipeline
from src.data_loader import load_label_mappings
from inference import align_labels

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your custom NER model and tokenizer
MODEL_DIR = "models/saved_ner_model"
ner_pipeline = pipeline("ner", model=MODEL_DIR)

# Load id_to_label mapping from JSON
print("Loading label mappings...")
_, id_to_label = load_label_mappings(MODEL_DIR)

# Function to generate random colors
def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Create dynamic color mapping with B- and I- labels sharing the same color
color_mapping = {}
for label in id_to_label.values():
    # Extract the base label without B- or I-
    base_label = label[2:] if label.startswith(("B-", "I-")) else label
    if base_label not in color_mapping:
        color_mapping[base_label] = generate_random_color()
        
# Update the mapping for B- and I- labels to share the same color
color_mapping.update({
    label: color_mapping[base_label]
    for label in id_to_label.values() if label.startswith(("B-", "I-"))
    for base_label in color_mapping if label[2:] == base_label
})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "highlighted_text": "", "color_mapping": color_mapping})

@app.post("/", response_class=HTMLResponse)
async def process_text(request: Request, input_text: str = Form(...)):
    highlighted_text = ""
    ner_results = ner_pipeline(input_text)
    words, labels = align_labels(ner_results, id_to_label)
    
    current_entity_tokens = []
    current_entity_label = None
    
    for word, label in zip(words, labels):
        if label != 'O':  # Not an "Other"
            base_label = label[2:]  # Extract the base label
            color_class = base_label  # Use the base label as class for color
            
            if current_entity_label is None:  # New entity
                current_entity_label = color_class
                current_entity_tokens.append(word)
            else:  # End of the current entity
                highlighted_text += (
                    f'<span class="{current_entity_label}">{"".join(current_entity_tokens)} <span class="tag">({current_entity_label})</span></span> '
                )
                current_entity_tokens = [word]
                current_entity_label = color_class
        else:  # Label is "O"
            if current_entity_label is not None:
                highlighted_text += (
                    f'<span class="{current_entity_label}">{"".join(current_entity_tokens)} <span class="tag">({current_entity_label})</span></span> '
                )
                current_entity_tokens = []
                current_entity_label = None
            highlighted_text += f'{word} '
    
    # Finalize any leftover entity tokens
    if current_entity_tokens:
        highlighted_text += (
            f'<span class="{current_entity_label}">{"".join(current_entity_tokens)} <span class="tag">({current_entity_label})</span></span>'
        )
    
    return templates.TemplateResponse("index.html", {"request": request, "highlighted_text": highlighted_text, "color_mapping": color_mapping})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)