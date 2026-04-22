# app.py — Application Streamlit : Moussa Kouyaté en Soninké
# TP2 — Déploiement GuppyLM
# =============================================================
# Lance avec :  streamlit run app.py

import os
import torch
import streamlit as st
from model import SonikeTokenizer, GuppyLM, generate

# ── Configuration de la page ──────────────────────────────────
st.set_page_config(
    page_title="Moussa Kouyaté — Soninké LM",
    page_icon="🌾",
    layout="centered",
)

# ── Chargement du modèle (mis en cache) ───────────────────────
@st.cache_resource
def load_model():
    tokenizer  = SonikeTokenizer()
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model      = GuppyLM(vocab_size=tokenizer.vocab_size).to(device)

    # Priorité au modèle fine-tuné, sinon le modèle pré-entraîné
    for path in ['guppylm_moussa_finetuned.pt', 'guppylm_moussa_best.pt']:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            st.sidebar.success(f'✅ Modèle chargé : `{path}`')
            break
    else:
        st.warning('⚠️ Aucun fichier .pt trouvé. Lance d\'abord l\'entraînement.')

    return model, tokenizer, device

# ── En-tête ───────────────────────────────────────────────────
st.title('🌾 Moussa Kouyaté — Dialogue en Soninké')
st.caption('Modèle GuppyLM · Vocabulaire SIL 3 231 mots · Kayes, Mali')

st.info(
    '**👨\u200d🌾 Qui est Moussa ?** '
    'Grand cultivateur de Kayes (Mali). Il cultive le mil (*yillen*), '
    'les arachides (*tigaren*) et élève son troupeau (*guma*). '
    'Parlez-lui de son champ, du marché, de la pluie ou de sa famille !'
)

st.divider()

# ── Barre latérale : paramètres ───────────────────────────────
with st.sidebar:
    st.header('⚙️ Paramètres de génération')
    temperature = st.slider(
        '🌡️ Température', 0.1, 2.0, 0.8, 0.1,
        help='Faible = réponses prévisibles | Élevée = réponses créatives'
    )
    max_tokens = st.slider('📏 Longueur max (tokens)', 20, 150, 80, 10)

    st.divider()
    st.subheader('💡 Exemples de questions')
    exemples = [
        'Haayi, Moussa!',
        'Yillen wa naxa ba ke xaaxo?',
        'Kanmen wa riini ba ke siine?',
        'Manni culture an wa soxo?',
        'An denbaya wa laafin ba?',
        'Alla wa an teen deema ba?',
        'Manni sagesse wa an maxa teen golle ba?',
        'An nanu wa laafin ba?',
        'Saxanen wa naxa ba ke fane?',
    ]
    for ex in exemples:
        if st.button(ex, use_container_width=True):
            st.session_state['user_input'] = ex

# ── Chargement ────────────────────────────────────────────────
with st.spinner('🔄 Chargement du modèle GuppyLM…'):
    model, tokenizer, device = load_model()

# ── Zone de saisie ────────────────────────────────────────────
user_input = st.text_input(
    '💬 Posez une question à Moussa en soninké :',
    value=st.session_state.get('user_input', ''),
    placeholder='Ex: Moussa, yillen wa naxa ba ke xaaxo?',
    key='input_box',
)

col1, col2 = st.columns([1, 4])
with col1:
    send = st.button('Envoyer 🚀', type='primary')
with col2:
    if st.button('Effacer 🗑️'):
        st.session_state['user_input'] = ''
        st.session_state['history']    = []
        st.rerun()

# ── Initialiser l'historique ──────────────────────────────────
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ── Génération ────────────────────────────────────────────────
if send and user_input.strip():
    prompt = user_input.strip() + ' →'
    with st.spinner('🌾 Moussa réfléchit…'):
        reponse = generate(
            model, tokenizer, prompt,
            max_new=max_tokens,
            temperature=temperature,
            device=device,
        )
    st.session_state['history'].append({
        'q': user_input.strip(),
        'r': reponse.strip(),
    })
    st.session_state['user_input'] = ''

# ── Affichage de l'historique ─────────────────────────────────
if st.session_state['history']:
    st.subheader('💬 Conversation')
    for turn in reversed(st.session_state['history']):
        with st.chat_message('user'):
            st.write(turn['q'])
        with st.chat_message('assistant', avatar='🌾'):
            st.write(turn['r'] or '*(réponse vide — essayez une autre formulation)*')

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    'TP2 Soninké · GuppyLM Transformer · '
    'Vocabulaire SIL 3 231 mots · Moussa Kouyaté, Kayes (Mali)'
)
