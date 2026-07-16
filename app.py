
import streamlit as st
import sqlite3, os, time, json
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# PIPELINE — loaded once, cached for the whole session
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Build every heavy component ONCE and reuse across all queries.
    Streamlit reruns this file on every click; cache_resource prevents
    reloading the 120 MB embedding model each time.
    """
    from sentence_transformers import SentenceTransformer
    import faiss, numpy as np
    import ollama as _ollama

    embed_model = SentenceTransformer(
        "intfloat/multilingual-e5-base",
        device="cpu",
    )

    db_path = Path(__file__).parent / "hospital.db"
    conn    = sqlite3.connect(str(db_path), check_same_thread=False)
    _seed_db(conn)

    idx, chunk_map = _build_index(conn, embed_model)
    return embed_model, conn, idx, chunk_map, _ollama


# ── Database ──────────────────────────────────────────────────────
def _seed_db(conn):
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS doctors(
        id INTEGER PRIMARY KEY, name TEXT, dept TEXT, lang TEXT, bio TEXT)""")
    cur.executemany("INSERT OR IGNORE INTO doctors VALUES(?,?,?,?,?)", [
        (1,"Dr. Amara Perera","Cardiology","en",
         "Dr. Amara Perera is a senior cardiologist with 20 years of experience. "
         "She specialises in interventional cardiology and heart failure management. "
         "Clinic: Monday, Wednesday, Friday 9am-1pm. Phone: 0112-345-001. Floor 3."),
        (2,"Dr. Suresh Kumar","Neurology","en",
         "Dr. Suresh Kumar leads the Neurology department. "
         "He treats epilepsy, stroke, Parkinson's disease, and migraines. "
         "Clinic: Tuesday and Thursday 10am-2pm. Phone: 0112-345-002. Floor 3."),
        (3,"Dr. Nimali Fernando","Paediatrics","en",
         "Dr. Nimali Fernando is a paediatrician caring for children from birth to 18. "
         "Vaccines, growth monitoring, asthma, and childhood illnesses. "
         "Appointments: weekdays 8am-12pm. Floor 2. Phone: 0112-345-003."),
        (4,"Dr. Ravi Jayasinghe","Orthopaedics","en",
         "Dr. Ravi Jayasinghe specialises in joint replacement and sports injuries. "
         "Knee and hip replacement, shoulder surgery. "
         "Wednesday and Saturday 9am-3pm. Floor 4. Phone: 0112-345-004."),
        (5,"Dr. Priya Balasingham","Gynaecology","en",
         "Dr. Priya Balasingham provides maternal and women health services. "
         "Antenatal care, family planning, menopause management. "
         "Mon-Fri 9am-5pm. Floor 2. Phone: 0112-345-005."),
        (6,"Dr. Chamara Wijeratne","Oncology","en",
         "Dr. Chamara Wijeratne is head of Oncology. "
         "Treats breast cancer, lung cancer, and blood cancers. "
         "Tuesdays and Thursdays 9am-1pm. Floor 5. Phone: 0112-345-006."),
        (7,"Dr. Lakmini Dias","Endocrinology","en",
         "Dr. Lakmini Dias manages diabetes, thyroid disorders, and hormonal conditions. "
         "Clinic: Mondays and Fridays 10am-2pm. Floor 3. Phone: 0112-345-007."),
        (8,"Dr. Kamala Dissanayake","General Medicine","si",
         "Dr. Kamala Dissanayake සාමාන්‍ය වෛද්‍ය විශේෂඥ. "
         "ඇය දියවැඩියාව, රුධිර පීඩනය, උණ සහ සාමාන්‍ය රෝග සඳහා ප්‍රතිකාර කරයි. "
         "වේලාව: සඳුදා සිට සිකුරාදා 8am-4pm. 1 වන මහල. දු.අ: 0112-345-008."),
        (9,"Dr. Vijay Arulrajah","Dermatology","ta",
         "Dr. Vijay Arulrajah தோல் நோய் நிபுணர். "
         "சொரியாஸிஸ், அரிப்பு, முகப்பரு மற்றும் தோல் புற்றுநோய் சிகிச்சை வழங்குகிறார். "
         "நேரம்: திங்கள், புதன், வெள்ளி 9am-1pm. இரண்டாவது மாடி. தொலைபேசி: 0112-345-009."),
        (10,"Dr. Ananthi Sivakumar","Cardiology","ta",
         "Dr. Ananthi Sivakumar இதய நோய் நிபுணர். "
         "மாரடைப்பு, இதய செயலிழப்பு மற்றும் உயர் இரத்த அழுத்தம் சிகிச்சை செய்கிறார். "
         "கிளினிக்: செவ்வாய், வியாழன் 9am-1pm. மூன்றாவது மாடி. தொலைபேசி: 0112-345-010."),
    ])

    cur.execute("""CREATE TABLE IF NOT EXISTS departments(
        id INTEGER PRIMARY KEY, name TEXT, floor INTEGER, phone TEXT, hours TEXT)""")
    cur.executemany("INSERT OR IGNORE INTO departments VALUES(?,?,?,?,?)", [
        (1,"Cardiology",3,"0112-345-001","Mon-Fri 8am-5pm"),
        (2,"Neurology",3,"0112-345-002","Mon-Fri 9am-4pm"),
        (3,"Paediatrics",2,"0112-345-003","Mon-Sat 8am-6pm"),
        (4,"Orthopaedics",4,"0112-345-004","Mon-Fri 8am-5pm"),
        (5,"Gynaecology",2,"0112-345-005","Mon-Fri 9am-5pm"),
        (6,"Oncology",5,"0112-345-006","Tue-Thu 9am-1pm"),
        (7,"Endocrinology",3,"0112-345-007","Mon, Fri 10am-2pm"),
        (8,"General Medicine",1,"0112-345-008","Mon-Sat 7am-7pm"),
        (9,"Dermatology",2,"0112-345-009","Mon-Fri 9am-4pm"),
        (10,"Emergency",0,"0112-345-911","24 hours / 7 days"),
        (11,"Pharmacy",1,"0112-345-010","Mon-Sun 7am-10pm"),
        (12,"Radiology",1,"0112-345-011","Mon-Fri 8am-8pm"),
        (13,"Pathology Lab",1,"0112-345-012","Mon-Sat 6am-6pm"),
        (14,"ICU",4,"0112-345-013","24 hours — no direct calls"),
    ])

    cur.execute("""CREATE TABLE IF NOT EXISTS services(
        id INTEGER PRIMARY KEY, title TEXT, lang TEXT, detail TEXT)""")
    cur.executemany("INSERT OR IGNORE INTO services VALUES(?,?,?,?)", [
        (1,"Appointment Booking","en",
         "To book an appointment call 0112-345-000 or visit reception on Floor 1. "
         "Online booking: hospital.lk/appointments. Bring your National ID. "
         "Cancellations must be made 24 hours in advance."),
        (2,"Emergency Services","en",
         "24-hour Emergency Department is on the Ground Floor. Ambulance: 1990. "
         "The ER handles trauma, chest pain, stroke, breathing difficulty, unconsciousness, "
         "and all life-threatening conditions. Triage within 5 minutes of arrival."),
        (3,"Lab Tests","en",
         "Pathology Lab is on Floor 1, open Mon-Sat 6am-6pm. "
         "Fasting blood tests must be done before 10am. "
         "Results online within 24 hours at hospital.lk/results. "
         "Urine, stool, and culture tests available same day."),
        (4,"Visiting Hours","en",
         "General wards: 4pm-6pm weekdays; 10am-12pm and 4pm-6pm weekends. "
         "ICU: 10am-11am and 4pm-5pm daily, maximum 2 visitors per patient. "
         "Children under 12 not permitted in ICU or surgical wards."),
        (5,"Insurance and Billing","en",
         "We accept all major Sri Lankan health insurance. "
         "Cashless treatment: Ceylinco, AIA, Softlogic, Union Assurance. "
         "Billing office: Floor 1, Mon-Fri 8am-5pm. Phone: 0112-345-099."),
        (6,"Pharmacy","en",
         "Hospital pharmacy is on Floor 1, open Mon-Sun 7am-10pm. "
         "Prescriptions filled within 20 minutes. Generic medicines at reduced cost. "
         "Phone: 0112-345-010."),
        (7,"Radiology and Imaging","en",
         "X-ray, CT scan, MRI, and ultrasound available on Floor 1. "
         "CT and MRI require a doctor referral. "
         "X-ray results in 1 hour. MRI reports in 24 hours. Phone: 0112-345-011."),
        (8,"Parking","en",
         "Free parking for up to 2 hours for patients and visitors. "
         "Enter from Hospital Road South Gate. Paid parking: Rs. 100/hour thereafter. "
         "Disability bays at main entrance."),
        (9,"Patient Portal","en",
         "Register at hospital.lk/portal to view test results, book appointments, "
         "and download discharge summaries. "
         "Requires a valid email and mobile number used at registration."),
        # ── Sinhala ──────────────────────────────────────────────
        (10,"හදිසි සේවා","si",
         "හදිසි ගිලන් රථ අංකය: 1990. "
         "හෘදයාබාධ, ආඝාත රෝග ලක්ෂණ, දැඩි ලේ ගැලීම ඇතිවිට "
         "වහාම ගිලන් රථ ඇමතිය යුතුය. "
         "හදිසි ඒකකය 24 පැය, සතිය පුරා විවෘතව ඇත. ශූන්‍ය මහල."),
        (11,"හමුවීම් ලබා ගැනීම","si",
         "හමුවීමක් ලබා ගැනීම සඳහා 0112-345-000 අංකය අමතන්න "
         "හෝ 1 වන මහලේ reception ට ගොඩ ගන්න. "
         "ජාතික හැඳුනුම්පත රැගෙන එන්න. "
         "අන්තර්ජාල හමුවීම්: hospital.lk/appointments."),
        (12,"ළමා රෝග ශාඛාව","si",
         "ළමා රෝග ශාඛාව 2 වන මහලෙහි ඇත. "
         "Dr. Nimali Fernando ළමා රෝග විශේෂඥ. "
         "සඳුදා සිට සිකුරාදා 8am-12pm. "
         "දුරකථනය: 0112-345-003."),
        (13,"රසායනාගාර පරීක්ෂණ","si",
         "රසායනාගාරය 1 වන මහලෙහි ඇත. "
         "සඳුදා සිට සෙනසුරාදා 6am-6pm. "
         "නිරාහාර රුධිර පරීක්ෂණ 10am ට පෙර ගත යුතුය. "
         "ප්‍රතිඵල: hospital.lk/results. දුරකථනය: 0112-345-012."),
        (14,"ෆාමසිය","si",
         "ෆාමසිය 1 වන මහලෙහි ඇත. "
         "සඳුදා සිට ඉරිදා 7am-10pm. "
         "ප්‍රේෂකණ (prescriptions) විනාඩි 20 ක් ඇතුළත ලබා දෙයි. "
         "දුරකථනය: 0112-345-010."),
        # ── Tamil ────────────────────────────────────────────────
        (15,"அவசர சேவைகள்","ta",
         "அவசர ஆம்புலன்ஸ்: 1990. "
         "மாரடைப்பு, பக்கவாதம், அதிரடி காயம், உணர்விழப்பு ஆகியவற்றிற்கு "
         "உடனடியாக அழைக்கவும். "
         "அவசர பிரிவு 24 மணி நேரமும், 7 நாட்களும் திறந்திருக்கும். தரை மாடி."),
        (16,"சந்திப்பு பதிவு","ta",
         "சந்திப்பு பதிவு செய்ய 0112-345-000 என்ற எண்ணில் அழைக்கவும் "
         "அல்லது முதல் மாடியில் உள்ள வரவேற்பு மையத்தை சந்திக்கவும். "
         "தேசிய அடையாள அட்டை கொண்டு வரவும். "
         "ஆன்லைன் பதிவு: hospital.lk/appointments."),
        (17,"மருந்தகம்","ta",
         "மருந்தகம் முதல் மாடியில் உள்ளது. "
         "திங்கள் முதல் ஞாயிறு வரை 7am-10pm. "
         "மருத்துவர் பரிந்துரை மருந்துகள் 20 நிமிடத்தில் வழங்கப்படும். "
         "தொலைபேசி: 0112-345-010."),
        (18,"ஆய்வக பரிசோதனைகள்","ta",
         "பரிசோதனை ஆய்வகம் முதல் மாடியில் உள்ளது. "
         "திங்கள் முதல் சனிக்கிழமை 6am-6pm. "
         "உண்ணாவிரத இரத்த பரிசோதனைகள் காலை 10 மணிக்கு முன். "
         "முடிவுகள்: hospital.lk/results. தொலைபேசி: 0112-345-012."),
        (19,"தோல் நோய் சிகிச்சை","ta",
         "Dr. Vijay Arulrajah தோல் நோய் நிபுணர், இரண்டாவது மாடி. "
         "சொரியாஸிஸ், அரிப்பு, முகப்பரு, தோல் புற்றுநோய் சிகிச்சை. "
         "திங்கள், புதன், வெள்ளி 9am-1pm. தொலைபேசி: 0112-345-009."),
    ])
    conn.commit()


# ── FAISS Index ───────────────────────────────────────────────────
def _build_index(conn, embed_model):
    import faiss, numpy as np

    rows = []
    cur = conn.cursor()
    for doc_id, name, dept, lang, bio in cur.execute(
            "SELECT id, name, dept, lang, bio FROM doctors"):
        rows.append({"source":"doctors","lang":lang,
                     "text": f"Doctor Profile — {name} ({dept}): {bio}"})

    for d_id, name, floor, phone, hours in cur.execute(
            "SELECT id, name, floor, phone, hours FROM departments"):
        rows.append({"source":"departments","lang":"en",
                     "text": f"Department {name}: Floor {floor}, Phone {phone}, Hours {hours}."})

    for s_id, title, lang, detail in cur.execute(
            "SELECT id, title, lang, detail FROM services"):
        rows.append({"source":"services","lang":lang,
                     "text": f"{title}: {detail}"})

    texts = [r["text"] for r in rows]
    vecs  = embed_model.encode(
        texts, batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    return idx, rows


# ── Language detection ────────────────────────────────────────────
def _detect_lang(text: str) -> str:
    """Detect script by Unicode code-point range. No ML needed."""
    for ch in text:
        cp = ord(ch)
        if 0x0D80 <= cp <= 0x0DFF: return "si"   # Sinhala
        if 0x0B80 <= cp <= 0x0BFF: return "ta"   # Tamil
    return "en"

_LANG_NAMES = {"en": "English", "si": "Sinhala", "ta": "Tamil"}


# ── Retrieval ─────────────────────────────────────────────────────
def _retrieve(query, embed_model, faiss_idx, chunk_map,
              top_k=6, threshold=0.30, lang_boost=0.05):
    import numpy as np

    user_lang = _detect_lang(query)
    q_vec = embed_model.encode(
        [query], normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_idx.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunk_map):
            continue
        chunk = chunk_map[idx].copy()
        adj   = float(score)
        if chunk.get("lang") == user_lang:
            adj += lang_boost          # reward same-language chunks
        if adj >= threshold:
            chunk["score"] = adj
            results.append(chunk)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results, user_lang


# ── Prompt builder — strict single-language, single-answer ────────
def _build_prompt(query: str, chunks: list, lang: str) -> str:
    """
    Forces the LLM to:
      1. Answer in exactly the same language as the question.
      2. Give ONE concise answer — not multiple options.
      3. Use only the provided context — no hallucination.
    """
    lang_name = _LANG_NAMES.get(lang, "English")

    lang_rules = {
        "en": "Reply in clear English. Be concise.",
        "si": (
            "ඔබ පිළිතුර සිංහලෙන් පමණක් ලබා දෙන්න. "
            "ඉංග්‍රීසියට මාරු නොවන්න. "
            "නිවැරදි තොරතුරු පමණක් ලබා දෙන්න."
            "අලුත් තොරතුරු සකසන්න එපා."
            "වෛද්‍ය නාම (medical terms) ඉංග්‍රීසියෙන් ලිවිය හැකිය. "
            "කෙටිව හා පැහැදිලිව ලිවිය යුතුය."
        ),
        "ta": (
            "நீங்கள் தமிழில் மட்டுமே பதில் அளிக்க வேண்டும். "
            "ஆங்கிலத்திற்கு மாறாதீர்கள். "
            "மருத்துவ சொற்கள் ஆங்கிலத்தில் இருக்கலாம். "
            "சுருக்கமாக பதில் அளிக்கவும்."
        ),
    }

    context_block = " \n\n ".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )

    return f"""You are a hospital information assistant.

STRICT RULES — follow all of them:
1. You MUST reply ONLY in {lang_name}. Do NOT switch to any other language.
2. Give exactly ONE direct answer. Do not list multiple options.
3. Use ONLY the information in the CONTEXT below. Do not guess or invent facts.
4. If the answer is not in the context, say (in {lang_name}):
   "I don't have that information. Please call reception: 0112-345-000."
5. {lang_rules.get(lang, lang_rules["en"])}

--- CONTEXT ---
{context_block}
--- END CONTEXT ---

Question: {query}

Answer (in {lang_name} only):"""


# ── LLM call ──────────────────────────────────────────────────────
def _ask_ollama(prompt: str, ollama, model: str) -> tuple:
    t0 = time.time()
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature" : 0.1,   # near-zero = deterministic, factual
                "num_predict" : 400,   # cap tokens → faster on CPU
                "num_ctx"     : 2048,
                "num_thread"  : 8,
            },
        )
        return resp["message"]["content"].strip(), time.time() - t0, None
    except Exception as e:
        return None, time.time() - t0, str(e)


# ══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Hospital AI | රෝහල් AI | மருத்துவமனை AI",
        page_icon="🏥",
        layout="wide",
    )

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        model = st.selectbox(
            "Ollama Model",
            ["llama3.2:1b", "llama3.2", "gemma3:12b", "qwen3:4b", "phi3"],
            help="llama3.2:1b is fastest on CPU (~800 MB).",
        )
        top_k = st.slider("Context chunks retrieved", 2, 8, 5)
        st.divider()
        st.markdown("**Languages**")
        st.markdown("🇬🇧 English &nbsp; 🇱🇰 සිංහල &nbsp; 🇮🇳 தமிழ்",
                    unsafe_allow_html=True)
        st.divider()
        st.markdown("**Stack**")
        st.markdown("""
        🦙 Ollama (local LLM)

        🔍 FAISS vector search

        🗄️ SQLite knowledge base
        """)
        if st.button("🗑️ Clear chat"):
            st.session_state.history = []
            st.rerun()

    # ── Header ───────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;color:#1a6fa1;'>"
        "🏥 Hospital Multilingual AI Assistant</h1>"
        "<p style='text-align:center;color:#666;'>"
        "Ask in <b>English</b>, <b>සිංහල</b>, or <b>தமிழ்</b> — "
        "answered in the <b>same language</b> using local AI</p><hr/>",
        unsafe_allow_html=True,
    )

    # ── Load pipeline ─────────────────────────────────────────────
    with st.spinner("Loading AI models… (first run ~30 s)"):
        try:
            embed_model, conn, faiss_idx, chunk_map, ollama = load_pipeline()
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.stop()
    st.success("✅ Ready! Type a question below.", icon="🚀")

    # ── Sample questions ──────────────────────────────────────────
    st.markdown("#### 💡 Sample questions — click to try:")
    cols = st.columns(3)
    samples = [
        ("🇬🇧 English",  ["Who is the cardiologist?",
                          "What time does the pharmacy close?",
                          "How do I book an appointment?"]),
        ("🇱🇰 සිංහල",  ["හදිසි ගිලන් රථ අංකය කුමක්ද?",
                          "ෆාමසිය කීයට වහනවාද?"]),
        ("🇮🇳 தமிழ்",  ["அவசர சேவை எண் என்ன?",
                          "மருந்தகம் எந்த நேரம் திறக்கும்?"]),
    ]
    chosen = None
    for col, (label, qs) in zip(cols, samples):
        with col:
            st.markdown(f"**{label}**")
            for q in qs:
                if st.button(q, key=q, use_container_width=True):
                    chosen = q

    st.divider()

    # ── Chat history ──────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []   # list of (role, text)

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    # Accept typed or button-clicked query
    typed = st.chat_input(
        "Ask your question… / ඔබගේ ප්‍රශ්නය… / உங்கள் கேள்வி…"
    )
    query = typed or chosen
    if not query:
        return

    # Show user bubble
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.history.append(("user", query))

    # ── Generate answer ───────────────────────────────────────────
    with st.chat_message("assistant"):
        lang = _detect_lang(query)
        lang_label = {"en":"🇬🇧 English","si":"🇱🇰 Sinhala","ta":"🇮🇳 Tamil"}.get(lang,"?")

        with st.spinner(f"Searching knowledge base ({lang_label})…"):
            chunks, _ = _retrieve(query, embed_model, faiss_idx, chunk_map, top_k)

        if not chunks:
            no_info = {
                "en": "I don't have that information. Please call reception: 0112-345-000.",
                "si": "ඒ තොරතුරු මා සතු නැත. කරුණාකර 0112-345-000 අමතන්න.",
                "ta": "அந்த தகவல் என்னிடம் இல்லை. 0112-345-000 என்ற எண்ணில் அழைக்கவும்.",
            }
            answer = no_info.get(lang, no_info["en"])
            st.markdown(answer)
        else:
            with st.expander(
                f"📄 {len(chunks)} relevant chunk(s) retrieved (click to inspect)"
            ):
                for i, c in enumerate(chunks, 1):
                    st.markdown(
                        f"**[{i}]** `{c['source']}` · score `{c['score']:.3f}`\n\n"  

                        f"{c['text'][:180]}{'…' if len(c['text'])>180 else ''}"
                    )

            prompt = _build_prompt(query, chunks, lang)

            with st.spinner("Generating answer with local LLM…"):
                t0 = time.time()
                answer, elapsed, err = _ask_ollama(prompt, ollama, model)

            if err:
                answer = (
                    f"⚠️ LLM error: {err} \n\n"


                    "Make sure Ollama is running: open a terminal and run `ollama serve`"
                )
                st.error(answer)
            else:
                st.markdown(answer)
                st.caption(
                    f"⏱ {elapsed:.1f}s · model: `{model}` · "
                    f"lang detected: `{lang}` · chunks used: {len(chunks)}"
                )

    st.session_state.history.append(("assistant", answer))

    # ── DB Explorer ───────────────────────────────────────────────
    with st.expander("🗄️ Hospital Database Explorer"):
        import pandas as pd
        t1, t2, t3 = st.tabs(["Doctors", "Departments", "Services"])
        cur = conn.cursor()
        with t1:
            st.dataframe(pd.read_sql(
                "SELECT id,name,dept,lang FROM doctors", conn),
                use_container_width=True)
        with t2:
            st.dataframe(pd.read_sql(
                "SELECT * FROM departments", conn),
                use_container_width=True)
        with t3:
            st.dataframe(pd.read_sql(
                "SELECT id,title,lang,detail FROM services", conn),
                use_container_width=True)


if __name__ == "__main__":
    main()
