# ==============================================================================
# 📝 사전 준비: 라이브러리 임포트 및 API 키 설정
# ==============================================================================
import google.generativeai as genai
import pandas as pd
import json
import os
import time
import re
import logging # 🪵 로깅 라이브러리 임포트

# ✨ 경로 설정은 그대로 유지합니다.
PATH = './data/dongwon/'
os.makedirs(PATH, exist_ok=True)

# 🪵 ======================= [신규] 로거 설정 ======================= 🪵
# 파일명에 사용할 타임스탬프를 먼저 생성
timestamp = time.strftime("%Y%m%d_%H%M%S")

# 로거 인스턴스 생성
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 포맷터 생성
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 스트림 핸들러 (콘솔 출력)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 파일 핸들러 (파일 출력)
log_filename = os.path.join(PATH, f'v3_log_{timestamp}.log')
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("✅ 로깅 설정이 완료되었습니다. 콘솔과 파일에 로그가 기록됩니다.")
# 🪵 =============================================================== 🪵

# [중요] 사용자의 API 키를 입력하세요.
try:
    # -------------------------------------------------------------------------
    genai.configure(api_key="") # <--- 여기에 실제 API 키를 입력하세요.
    # -------------------------------------------------------------------------
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    logger.info("✅ Gemini API 키가 설정되었습니다.")
except Exception as e:
    logger.error(f"❗️ API 키 설정 중 오류가 발생했습니다: {e}")


# 헬퍼 함수 (기존과 동일)
def extract_json_from_response(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# ==============================================================================
# ✨ 1. [신규] 제품별 페르소나 생성을 위한 동적 프롬프트 생성 함수
# ==============================================================================
def create_product_specific_prompt(product_name, num_personas=30): # 🪵 기본값 30으로 수정
    """
    제품 이름과 특성을 기반으로 맞춤형 페르소나 생성 프롬프트를 생성합니다.
    """
    target_customer_profile = "일반적인 대한민국 소비자" # 기본값

    # --- 제품군별 타겟 고객 프로필 정의 ---
    if '하이그릭요거트' in product_name:
        target_customer_profile = "20-40대 여성으로, 건강과 자기관리에 관심이 많고, 소득 수준은 중상 이상인 1인 가구. 주로 온라인 채널을 통해 건강식품을 구매하는 경향이 있음."
    elif '맛참' in product_name:
        target_customer_profile = "10-30대 남녀로, 편의성과 새로운 맛을 추구하며 유튜브, 인스타그램 등 소셜 미디어에 익숙함. 편의점이나 온라인에서 간편하게 식사를 해결하려는 1인 가구 학생 및 직장인."
        if '매콤' in product_name:
            target_customer_profile += " 특히 매운맛을 선호하는 경향이 뚜렷함."
    elif '리챔' in product_name:
        target_customer_profile = "30-40대 주부로, 3인 이상 가구의 식사를 책임지고 있음. 대형마트에서 장을 보며, 명절 등 특별한 날에 가족을 위한 요리를 준비하는 것을 중요하게 생각함."
    elif '참치액' in product_name:
        target_customer_profile = "요리에 관심이 많은 30-50대 주부 또는 1인 가구. 집밥을 선호하며, 음식의 깊은 맛을 내기 위한 조미료에 투자를 아끼지 않음. 대형마트와 온라인 채널을 모두 이용함."
        if '진' in product_name or '프리미엄' in product_name:
            target_customer_profile += " 특히 요리 실력이 뛰어나고, 소득 수준이 높아 프리미엄 제품을 선호하는 미식가적 성향을 보임."
    elif '소화가 잘되는' in product_name:
        target_customer_profile = "유당불내증이 있거나 소화 건강에 신경 쓰는 20-50대 직장인. 건강을 위해 일반 유제품 대신 락토프리 제품을 선택하며, 출근길이나 점심시간에 편의점에서 자주 구매함."
        if '바닐라라떼' in product_name:
            target_customer_profile += " 단맛을 선호하는 젊은 층의 비중이 상대적으로 높음."

    # 🪵 타겟 프로필을 로그로 기록
    logger.info(f"  - 타겟 프로필 설정: {target_customer_profile}")

    prompt = f"""
    당신은 특정 제품의 핵심 구매 고객 페르소나를 생성하는 마케팅 분석 AI입니다.

    [지시사항]
    1.  **임무**: 아래 [제품 정보]에 명시된 제품을 구매할 가능성이 매우 높은 **핵심 고객 페르소나 {num_personas}개**를 생성합니다.
    2.  **핵심 조건**: 생성되는 페르소나는 아래 [타겟 고객 프로필]의 특성을 집중적으로 반영해야 합니다. 페르소나의 모든 속성은 이 프로필과 논리적으로 강력하게 연결되어야 합니다.
    3.  **출력**: 다른 설명 없이, [출력 형식 예시]를 완벽히 따르는 단일 JSON 배열만 반환하세요.

    [제품 정보]
    - 제품명: "{product_name}"

    [타겟 고객 프로필]
    - {target_customer_profile}

    [속성 가이드]
    - `age`: "10대", "20대", "30대", "40대", "50대", "60대 이상"
    - `gender`: "남성", "여성"
    - `occupation`: "직장인", "학생", "주부", "프리랜서", "자영업자", "무직", "은퇴"
    - `income_level`: "상", "중상", "중", "중하", "하"
    - `household_size`: "1인 가구", "2인 가구", "3인 가구", "4인 가구 이상"
    - `lifestyle`: "건강지향", "자기관리", "편의성추구", "가성비중시", "트렌드추구", "요리애호가", "집밥선호", "미식가"
    - `media_consumption`: ["YouTube", "Instagram", "TV", "커뮤니티사이트"] 등
    - `price_sensitivity`: "높음", "중간", "낮음"
    - `brand_loyalty`: "높음", "중간", "낮음"
    - `dietary_preferences`: "고단백", "저칼로리", "유당불내증케어", "소화편한음식선호", "매운맛선호" 등
    - `shopping_channel`: "대형마트", "온라인", "편의점", "백화점"

    [출력 형식 예시]
    [
      {{
        "persona_id": "P00001", "attributes": {{"age": "30대", "gender": "여성", "occupation": "직장인", "income_level": "중상", "household_size": "1인 가구", "lifestyle": "건강지향", "media_consumption": ["YouTube", "Instagram"], "price_sensitivity": "중간", "brand_loyalty": "낮음", "dietary_preferences": "고단백", "shopping_channel": "온라인"}}, "purchase_probability": 0.85, "base_purchase_frequency_per_month": 4.0, "monthly_propensity_modifier": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      }}
    ]
    """
    return prompt

logger.info("✅ 1. 제품별 동적 프롬프트 생성 함수가 준비되었습니다.")

# 🪵 ======================= [신규] 페르소나 분석 로그 함수 ======================= 🪵
def log_persona_summary(df, product_name):
    if df.empty:
        return
    logger.info(f"--- [ {product_name} ] 페르소나 분석 결과 ---")
    logger.info(f"  - 총 {len(df)}개의 페르소나 생성됨")
    
    # 구매 행동 관련 로그
    avg_prob = df['purchase_probability'].mean()
    avg_freq = df['base_purchase_frequency_per_month'].mean()
    logger.info(f"  - 평균 구매 확률: {avg_prob:.2f}")
    logger.info(f"  - 평균 월 구매 빈도: {avg_freq:.2f}")

    # 주요 속성 분포 로그 (상위 3개)
    key_attributes = ['age', 'lifestyle', 'shopping_channel', 'occupation']
    for attr in key_attributes:
        if attr in df.columns:
            dist = df[attr].value_counts(normalize=True).nlargest(3) * 100
            dist_str = ", ".join([f"{idx} {val:.1f}%" for idx, val in dist.items()])
            logger.info(f"  - '{attr}' 분포: {dist_str}")
    logger.info("----------------------------------------------------")
# 🪵 =========================================================================== 🪵

# ==============================================================================
# ✨ 2. 시뮬레이션 파라미터 정의 (기존과 동일)
# ==============================================================================
HOLIDAY_MODIFIERS = {
    'seollal_chuseok': [1.8, 5.5, 1.0, 0.9, 1.0, 1.1, 1.2, 2.5, 6.5, 1.0, 1.0, 1.1],
    'seollal_chuseok_sub': [1.3, 3.0, 1.0, 1.1, 1.2, 1.1, 1.2, 1.8, 3.5, 1.1, 1.0, 1.3]
}
DEFAULT_MODIFIERS = [1.0] * 12
SIMULATION_PARAMS = {
    '덴마크 하이그릭요거트 400g': {'tam': 600000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.2, 1.4, 1.4, 1.3, 1.2, 1.2, 0.9, 0.8, 0.7, 0.7]},
    '동원맛참 고소참기름 135g': {'tam': 2000000, 'market_share': 0.07, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 고소참기름 90g':  {'tam': 2000000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 매콤참기름 135g': {'tam': 2000000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 매콤참기름 90g':  {'tam': 2000000, 'market_share': 0.03, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '리챔 오믈렛햄 200g': {'tam': 3000000, 'market_share': 0.3, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok']},
    '리챔 오믈렛햄 340g': {'tam': 3000000, 'market_share': 0.5, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok']},
    '동원참치액 순 500g':  {'tam': 250000, 'market_share': 0.025, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 순 900g':  {'tam': 250000, 'market_share': 0.020, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 진 500g':  {'tam': 250000, 'market_share': 0.030, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 진 900g':  {'tam': 250000, 'market_share': 0.025, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '프리미엄 동원참치액 500g': {'tam': 250000, 'market_share': 0.015, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '프리미엄 동원참치액 900g': {'tam': 250000, 'market_share': 0.010, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '소화가 잘되는 우유로 만든 바닐라라떼 250mL': {'tam': 2000000, 'market_share': 0.09, 'modifiers': [1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1]},
    '소화가 잘되는 우유로 만든 카페라떼 250mL':   {'tam': 2000000, 'market_share': 0.12, 'modifiers': [1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1]},
    'default': {'tam': 1000000, 'market_share': 0.10, 'modifiers': DEFAULT_MODIFIERS}
}
START_MONTH_INDEX = 6
for product, params in SIMULATION_PARAMS.items():
    original_modifiers = params['modifiers']
    reordered_modifiers = original_modifiers[START_MONTH_INDEX:] + original_modifiers[:START_MONTH_INDEX]
    SIMULATION_PARAMS[product]['modifiers'] = reordered_modifiers

logger.info("✅ 2. SKU별 시뮬레이션 파라미터 설정 및 월별 가중치 재정렬이 완료되었습니다.")


# ==============================================================================
# ✨ 3. 제품별 페르소나 생성 및 판매량 시뮬레이션 (로깅 강화)
# ==============================================================================
try:
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    logger.info("✅ 3. 제출용 데이터프레임 로드를 완료했습니다. 이제 시뮬레이션을 시작합니다.")
except FileNotFoundError as e:
    logger.error(f"❗️ 파일 로드 실패: {e}. 'sample_submission.csv' 파일이 있는지 확인해주세요.")
    exit()

# 시뮬레이션 설정
PERSONAS_PER_BATCH = 30
NUM_BATCHES_PER_PRODUCT = 4
MAX_RETRIES_PER_BATCH = 4
submission_filename = os.path.join(PATH, f'my_submission_v3_{timestamp}.csv')

# --- 메인 시뮬레이션 루프 ---
for index, row in submission_df.iterrows():
    product_name = row['product_name']
    logger.info(f"\n==================== [ {product_name} ] 판매량 예측 시작 ====================")

    params = SIMULATION_PARAMS.get(product_name, SIMULATION_PARAMS['default'])
    
    product_personas = []
    for i in range(NUM_BATCHES_PER_PRODUCT):
        for attempt in range(MAX_RETRIES_PER_BATCH):
            try:
                prompt = create_product_specific_prompt(product_name, PERSONAS_PER_BATCH)
                logger.info(f" ⏳ 배치 {i+1}/{NUM_BATCHES_PER_PRODUCT} API 호출 중... (시도 {attempt+1})")
                response = model.generate_content(prompt)
                
                json_text = extract_json_from_response(response.text)
                if not json_text:
                    raise ValueError("응답에서 JSON을 찾을 수 없습니다.")
                
                batch_personas = json.loads(json_text)
                product_personas.extend(batch_personas)
                logger.info(f" ✅ 배치 {i+1} 생성 완료! ({len(batch_personas)}명 추가)")
                time.sleep(20)
                break

            except Exception as e:
                logger.warning(f" ❗️ 배치 {i+1} 시도 {attempt+1} 실패: {e}")
                if attempt < MAX_RETRIES_PER_BATCH - 1:
                    logger.info(" 20초 후 재시도합니다...")
                    time.sleep(20)
                else:
                    logger.error(f" ❌ 배치 {i+1} 생성 최종 실패.")
    
    if not product_personas:
        logger.error(f" 🚫 페르소나 생성에 실패하여 [ {product_name} ]의 판매량을 0으로 처리합니다.")
        submission_df.iloc[index, 1:] = [0] * 12
        continue

    personas_df_product = pd.DataFrame(product_personas)
    attributes_df_product = pd.json_normalize(personas_df_product['attributes'])
    personas_df_product = pd.concat([personas_df_product.drop('attributes', axis=1), attributes_df_product], axis=1)

    # 🪵 생성된 페르소나의 특성을 로그로 기록
    log_persona_summary(personas_df_product, product_name)

    logger.info(f"--- [ {product_name} ] 월별 판매량 계산 시작 ---")
    monthly_sales = []
    num_personas = len(personas_df_product)
    
    # 예측 기간(7월~6월)에 맞춰 월 이름 리스트 생성
    month_labels = [f"{m}월" for m in list(range(7, 13)) + list(range(1, 7))]

    for month_index in range(12):
        month_modifier = params['modifiers'][month_index]
        
        # 🪵 계산 과정을 단계별로 분해하여 로깅
        base_purchase_points = (
            personas_df_product['purchase_probability'] *
            personas_df_product['base_purchase_frequency_per_month']
        )
        total_purchase_points = (base_purchase_points * month_modifier).sum()
        
        avg_points_per_persona = total_purchase_points / num_personas if num_personas > 0 else 0
        
        final_sales = avg_points_per_persona * params['tam'] * params['market_share']
        monthly_sales.append(int(final_sales))
        
        # 🪵 계산 로그 출력
        log_msg = (
            f"  - [{month_labels[month_index]:>3s}] 계산: "
            f"페르소나 평균 구매력({avg_points_per_persona / month_modifier:.2f}) "
            f"x 월 계수({month_modifier:.2f}) "
            f"x TAM({params['tam']}) "
            f"x 점유율({params['market_share']:.3f}) "
            f"= 최종 판매량: {int(final_sales)}"
        )
        logger.info(log_msg)

    submission_df.iloc[index, 1:] = monthly_sales
    logger.info(f"📈 [ {product_name} ] 12개월 판매량 예측 완료!")

    if index < len(submission_df) - 1:
        logger.info("🕒 다음 제품 분석 전, API 할당량 준수를 위해 60초간 대기합니다...")
        time.sleep(60)

# --- 최종 파일 저장 ---
submission_df.to_csv(submission_filename, index=False)
logger.info(f"\n\n🎉🎉🎉 모든 제품의 시뮬레이션이 완료되었습니다!")
logger.info(f"✅ 최종 제출 파일 '{submission_filename}' 생성이 완료되었습니다.")
logger.info(f"✅ 상세 로그는 '{log_filename}' 파일에 저장되었습니다.")

print("\n[ 최종 예측 결과 (상위 5개 제품) ]")
print(submission_df.head())