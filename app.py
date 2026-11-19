import streamlit as st
from transformers import pipeline
import re

# 設定網頁標題與基本設定
st.set_page_config(page_title="情緒分析星等預測器", page_icon="📊")
st.title("🤖 產品評論情緒分析器")
st.write("請輸入產品評論或使用心得，系統會自動評定為 1 到 5 個星等。")

# 載入情緒分析模型 (使用快取避免每次執行重複載入)
@st.cache_resource
def load_model():
    try:
        return pipeline("sentiment-analysis", model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")
    except Exception as e:
        return None

classifier = load_model()

def analyze_sentiment_to_level(text):
    cleaned_text = text.strip()
    if not cleaned_text:
        return "請輸入有效的文字。"

    # 1. 強中性關鍵字優先判定
    neutral_indicators = ['普通', '一般般', '一般', '還好', '普通般', '沒感覺', '尚可', '過得去', '平平']
    for indicator in neutral_indicators:
        if indicator in cleaned_text:
            if not any(strong in cleaned_text for strong in ['非常', '超級', '無敵', '極度', '特地', '非常之']):
                return 3

    if classifier is not None:
        # 2. 模型預測與星等轉換
        result = classifier(cleaned_text)[0]
        label = result['label']
        score = result['score']
        
        is_positive = '1' in label or 'positive' in label.lower()
        
        if is_positive:
            if score > 0.85:
                return 5  
            elif score > 0.55:
                return 4  
            else:
                return 3  
        else:
            if score > 0.85:
                return 1  
            elif score > 0.55:
                return 2  
            else:
                return 3  

    # 3. 備用規則分析系統（僅在模型損壞時使用）
    positive_strong = ['超級好', '太棒', '非常滿意', '完美', '滿分', '極推', '無敵', '必買', '超讚']
    positive_weak = ['好', '讚', '棒', '滿意', '不錯', '推薦', '好吃', '好用', '划算', '漂亮', '喜歡']
    
    negative_strong = ['極差', '爛透', '非常失望', '垃圾', '別買', '超級爛', '雷到爆', '難吃到死', '不要買']
    negative_weak = ['差', '爛', '失望', '難用', '生氣', '雷', '糟糕', '難吃', '難看', '難聽', '不推', '不行', '差勁']

    if any(word in cleaned_text for word in positive_strong):
        return 5
    if any(word in cleaned_text for word in negative_strong):
        return 1
    
    pos_score = sum(1 for word in positive_weak if word in cleaned_text)
    neg_score = sum(1 for word in negative_weak if word in cleaned_text)
    
    if any(mod in cleaned_text for mod in ['太', '這麼', '非常', '特別', '真的很']) and neg_score > 0:
        return 1

    if pos_score > neg_score:
        return 4
    elif neg_score > pos_score:
        return 2
    else:
        return 3

# 建立使用者介面
user_input = st.text_area("💬 輸入您對產品的評論文字：", placeholder="例如：這個產品真的超級好用...")

if st.button("🎯 開始分析情緒"):
    if user_input.strip() == "":
        st.warning("⚠️ 請先輸入文字再進行分析。")
    else:
        with st.spinner("分析中，請稍候..."):
            rating = analyze_sentiment_to_level(user_input)
            
        # 呈現結果畫面
        st.subheader("📊 預測結果")
        stars = "⭐" * rating
        st.markdown(f"### 情緒評等：**{rating} 分** ({stars})")
        
        if rating >= 4:
            st.success("🎉 這是一則正面的好評！")
        elif rating == 3:
            st.info("😐 這是一則中性/普通的評價。")
        else:
            st.error("⚠️ 這是一則負面的差評！")
