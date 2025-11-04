# KDT_Web  
KDT 웹 프론트엔드/백엔드 통합 실습 프로젝트

## 🧩 프로젝트 개요  
이 프로젝트는 KDT 강의에서 웹 개발 종합 실습을 위해 진행된 저장소입니다.  
프론트엔드(React/HTML/CSS/JavaScript)와 백엔드(Node.js, Express 등)를 통합하여 실제와 유사한 웹 애플리케이션을 구현하는 것을 목표로 합니다.  
웹 서비스 전체 흐름을 경험하고, 클라이언트-서버 구조, REST API 통신, 인증/인가, 데이터 저장 및 UI 구현까지 폭넓게 다루었습니다.

## 🚀 주요 기능  
- 회원가입 / 로그인 / 로그아웃 기능 구현  
- CRUD 기능을 갖춘 데이터 관리 (게시글, 댓글, 파일 업로드 등)  
- RESTful API 설계 및 프론트엔드 연동  
- 반응형 웹 디자인 구현  
- 개발/배포 환경 분리 및 환경 변수 관리  
- (선택사항) 실시간 기능(웹소켓) 또는 외부 API 연동  

## 🛠 기술 스택  
| 구분 | 기술 |
|------|------|
| 프론트엔드 | HTML5, CSS3, JavaScript, React.js |
| 백엔드 | Node.js, Express |
| 데이터베이스 | MongoDB, MySQL, PostgreSQL 등 |
| 스타일링 | Sass, Styled Components, TailwindCSS |
| 인증/인가 | JWT, OAuth (선택) |
| 배포 및 도구 | Git, GitHub, Docker, AWS 또는 Heroku |

> **참고:** 실제 사용된 기술 스택은 `package.json` 또는 환경 설정 파일을 참고하세요.

## 📁 디렉토리 구조 예시  
```
/KDT_Web
├─ client # 프론트엔드 소스 코드
│ ├─ components # 공통 UI 컴포넌트
│ ├─ pages # 주요 페이지 (Home, Login, Signup 등)
│ ├─ assets # 이미지, 폰트, 아이콘 등 정적 자원
│ ├─ styles # 전역 스타일 및 CSS 파일
│ └─ App.js # 루트 컴포넌트
│
├─ server # 백엔드 소스 코드
│ ├─ routes # API 라우터 정의
│ ├─ models # 데이터베이스 모델
│ ├─ controllers # 비즈니스 로직
│ ├─ middlewares # 인증, 에러 핸들링 등 미들웨어
│ └─ app.js # 서버 진입점
│
├─ .env.example # 환경변수 예시 파일
└─ README.md # 프로젝트 설명 파일
```
