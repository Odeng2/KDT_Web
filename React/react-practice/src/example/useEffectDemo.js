import {useState, useEffect} from 'react';


const UseEffectDemo = () => {

    const [count, setCount] = useState(0);
    const [userInput, setUserInput] = useState({
        title: "",
        content:  "",
        date: ""
    })

    useEffect(() => {
        // 의존성 배열 안에 값이 변하메 따라 동작하는 함수
        console.log("ComponentDidMount");
    }, [])

    useEffect(() => {
        console.log("ComponentDidMount + ComponentDidUpdate");
    }, [])

    useEffect(() => {
        console.log("ComponentDidMount + ComponentDidUpdate(title만 변경시)");
    }, [userInput.title]);

    const handleClick = () => {
        setCount(count+1);
    }

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
};

export default UseEffectDemo;