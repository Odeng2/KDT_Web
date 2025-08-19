// store.js
import { configureStore } from "@reduxjs/toolkit";
import counterReducer from "../reducers/counterReducer";

// 스토어 생성
const store = configureStore({
    reducer: counterReducer   //필요한 reducer를 전부 담을 수 있음.
});

export default store;