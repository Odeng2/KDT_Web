
import './App.css';
// import React from 'react';



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




// //예시: 엘리먼트 렌더링, Virtual DOM

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
//         <li>사과바나나</li>
//         <li>사과바나나</li>
//         <li>사과바나나</li>
//         <li>사과바나나</li>
//       </ul>
//     </React.Fragment>
//   );
// }

// export default App;



// //8/14: Props

// props = {userName: "gildong"}

// function App() {
//   const listData = [
//     { imgURL: "./image1.jpg", title: "상품명1"},
//     { imgURL: "./image2.jpg", title: "상품명2"},
//     { imgURL: "./image3.jpg", title: "상품명3"},
//     { imgURL: "./image4.jpg", title: "상품명4"},
//     { imgURL: "./image5.jpg", title: "상품명5"},
//     { imgURL: "./image6.jpg", title: "상품명6"},
//     { imgURL: "./image7.jpg", title: "상품명7"},
//     { imgURL: "./image8.jpg", title: "상품명8"},
//   ]
  
//   return (
//     <div className='card-list'>

//       {/* /* 방법1 
//       <Card data={listData[0]}/>
//       <Card data={listData[1]}/>
//       <Card data={listData[2]}/>
//       <Card data={listData[3]}/>
//       <Card data={listData[4]}/>
//       <Card data={listData[5]}/>
//       <Card data={listData[6]}/>
//       <Card data={listData[7]}/> */}


//       {/* 방법2 */}
//       {
//         listData.map((item) =>
//           <Card kery={item.imgURL} data={data}/>
//         )
//       }
//     </div>
//   );
// }

// export default MainHeader;




// //state 업데이트 예시 코드

// import React, { Component } from "react";

// class NameChanger extends Component {
//   // constsructor: 클래스를 초기화함
//   constructor(props) {
//     super(props);
//     // 초기 상태 설정
//     this.state = {
//       name: "나리",
//     };
//   }

//   changeName = () => {
//     // // 잘못된 방법 (state를 직접 수정)
//     // this.state = {
//     //   name: "둘리",
//     // };

//     // 올바른 방법
//     this.setState({ 
//       name: "둘리",
//     });
//     console.log(this.state.name);
//   };

//   render() {
//     return (
//       <div>
//         <p>현재 이름: {this.state.name}</p>
//         <button onClick={this.changeName}>이름 변경</button>
//       </div>
//     );
//   }
// }

// export default NameChanger;



// //Lifecycle 예제

// import LifecycleDemo from "./example/lifecycleDemo";

// function App() {
//   return (
//     <LifecycleDemo/>
//   )
// }

// export default App;




//

// import HookDemo from './example/hookDemo';

// function App () {
//   return (
//     <HookDemo/>
//   )
// }

// export default App;

// import UseEffectDemo from './example/hookDemo';

// function App () {
//   return (
//     <UseEffectDemo/>
//   )
// }

// export default App;

// import React, { useState } from 'react';
// import ChildComponent from './example/childComponent';

// function App () {

//   const [ parentState, setParentState ] = useState("초기값");

//   const handleChangeFromParent = (dataFromChild) => {
//     console.log(dataFromChild);
//     setParentState(dataFromChild);
//   }

//   return (
//     <>
//       <h1>부모의 상태값 : {parentState}</h1>
//       <ChildComponent 
//       dataForChild={parentState}
//       onChange={handleChangeFromParent}/>
//     </>
    
//   );
// }

// export default App;



import React from 'react';
import CounterDemo from './components/examples/counterDemo';
import ShowCounterDemo from './components/examples/showCounterDemo';

function App () {

  return (
    <>
      <CounterDemo/>
      <ShowCounterDemo/>
    </>
  );
}

export default App;