프로젝트명:
Server Resource Action Recommendation using DQN

목표:
서버 리소스 상태(cpu, memory, temperature 등)를 기반으로 workload action을 추천하는 DQN 기반 의사결정 보조 모델 개발

시스템 구조
cpu / memory 점유율 상태를 통해 workload 자원 관리

현재 구현:
- custom env.py 구현
- DQN train.py 구현
- replay buffer / epsilon-greedy / target network 적용
- reward tuning 진행
- random policy / rule policy / DQN policy 비교
- model save/load 기능 추가

현재 결론:
현재 df 기반 환경에서는 action이 next_state에 직접 영향을 주지 않기 때문에,
모델은 평균적으로 가장 유리한 action으로 수렴하는 경향이 있다.
따라서 현재 버전은 실제 제어기보다는 로그 기반 action 추천 모델에 가깝다.

다음 목표:
실제 시스템에서 action이 workload에 영향을 주도록 stress/stress-ng(cpu or memory 부하 프로그램) 기반 제어 환경을 구현한다.

현재 상태 수집
→ 모델이 action 선택
→ action 실행
   0: hold
   1: decrease
   2: increase
→ 일정 시간 대기
→ 다음 상태 수집
→ reward 계산
→ 학습/평가

#실행 방법
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train
python3 /real/trin_real.py

#rule_based
python3 /real/rule_real_eval.py

#evalute
python3 /real/run_real_eval.py

실험 결과
rule_based 45
DQN 54

향후 개선 방향
현재 실험은 CPU workload 중심으로 진행되었으며, 향후 Memory 및 GPU workload까지 확장할 필요가 있다.

추가 dual CPU 및 multi-core 서버 환경 대응을 위해 CPU core 기반 정규화 및 workload scaling 기법을 적용할 예정이다.


