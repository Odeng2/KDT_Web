import {useState} from 'react';

const HookDemo = () => {
  // let count = 0;
  const [count, setCount] = useState(0);

  // const handleClick = () => {
  //   // 기본 형태: setState(currentState); prevState => CurrentState
  //   // 아래의 두 가지 방법 다 가능. 한 개만 주석 없애고 사용하기

  //   // setCount((count) => count+1);
  //   setCount((prevState) => prevState+1);
    
  // }


  // const [arrData, setArrData] = useState([1, 2, 3]);

  // const handleClick = () => {
  //   // // 잘못된 방식 #1
  //   // arrData.push(4);
  //   // setArrData(arrData);
  //   // // 잘못된 방식 #2
  //   // setArrData(arrData.push(4));

  //   // 올바른 방식
  //   setArrData([...arrData, 4]);
  // }

  const [userInput, setUserInput] = useState({
    title: "",
    content:  "",
    date: ""
  })

  const handleClick = () => {

  }


  // 권장 X 방법
  // const handleChange = (e) => {
  //   // userInput.title = e.target.value;
  //   setUserInput({
  //     ...userInput,
  //     title: e.target.value
  //   })
  // }
  


  // 권장하는 방법
  const handleChange = (e) => {
    // userInput.title = e.target.value;
    setUserInput((prevState) => {
      return {...prevState, title: e.target.value}
    })
  }
  

  return (
    <div>
      {/* <p>현재 카운트: {count}</p> */}
      <p>타이틀: {userInput.title}</p>
      <input 
      value={userInput.title}
      onChange={handleChange}
      />
      <button onClick={handleClick}>증가</button>
    </div>
  );
}

export default HookDemo;