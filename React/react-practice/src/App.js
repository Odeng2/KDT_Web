import logo from './logo.svg';
import './App.css';
import React from 'react';



// 수업 실습 #1

// function App() {
//   const arr1 = [
//     { id:1, name:"gildong", age:30},
//     { id:2, name:"gildong", age:30},
//     { id:3, name:"gildong", age:30},
//     { id:4, name:"gildong", age:30},
//     { id:5, name:"gildong", age:30},
//   ]
//   return (
//     <div className="header">
//       <h1 className="">Hello, React!</h1>
//       <h2 className="">Hello, React!</h2>
//       <ul>
//         {
//           arr1.map((arrItem) =>
//             <li key={arrItem.id}>{arrItem.name}</li>
//           )
//         }
//       </ul>
//     </div>
//   );
// }

// export default App;




////실습 #2: div대신 사용할 수 있는 태그: <>

// function App() {
//   const arr1 = [
//     { id:1, name:"gildong", age:30},
//     { id:2, name:"gildong", age:30},
//     { id:3, name:"gildong", age:30},
//     { id:4, name:"gildong", age:30},
//     { id:5, name:"gildong", age:30},
//   ]
//   return (
//     <>
//       <h1 className="">Hello, React!</h1>
//       <h2 className="">Hello, React!</h2>
//       <ul>
//         {
//           arr1.map((arrItem) =>
//             <li key={arrItem.id}>{arrItem.name}</li>
//           )
//         }
//       </ul>
//     </>
//   );
// }

// export default App;





////실습 #3: div대신 사용할 수 있는 태그 2: <React.Fragment>

// function App() {
//   const arr1 = [
//     { id:1, name:"gildong", age:30},
//     { id:2, name:"gildong", age:30},
//     { id:3, name:"gildong", age:30},
//     { id:4, name:"gildong", age:30},
//     { id:5, name:"gildong", age:30},
//   ]
//   return (
//     <React.Fragment>
//       <h1 className="">Hello, React!</h1>
//       <h2 className="">Hello, React!</h2>
//       <ul>
//         {
//           arr1.map((arrItem) =>
//             <li key={arrItem.id}>{arrItem.name}</li>
//           )
//         }
//       </ul>
//     </React.Fragment>
//   );
// }

// export default App;




//예시: 엘리먼트 렌더링, Virtual DOM

function App() {
  const arr1 = [
    { id:1, name:"gildong", age:30},
    { id:2, name:"gildong", age:30},
    { id:3, name:"gildong", age:30},
    { id:4, name:"gildong", age:30},
    { id:5, name:"gildong", age:30},
  ]
  return (
    <React.Fragment>
      <h1 className="">Hello, React!</h1>
      <h2 className="">Hello, React!</h2>
      <ul>
        <li>사과바나나</li>
        <li>사과바나나</li>
        <li>사과바나나</li>
        <li>사과바나나</li>
      </ul>
    </React.Fragment>
  );
}

export default App;