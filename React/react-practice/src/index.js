import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';    //CSS 파일 가져오기: 클론코딩할때는 App.js 활용
import App from './App';
// import reportWebVitals from './reportWebVitals';    //에러 코드. 기본 내장되어 있음. 없어도 괜찮다!

// // //index.html의 아래 코드와 같은 내용임.
// // ReactDOM.render(<App/>, document.getElementById("root"));
// const root = ReactDOM.createRoot(document.getElementById('root'));
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );

// // If you want to start measuring performance in your app, pass a function
// // to log results (for example: reportWebVitals(console.log))
// // or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// reportWebVitals();


//8.19: redux
import { Provider } from 'react-redux';
import store from './store/store';


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <Provider store={store}>
    <App />
  </Provider>
);