import { useSelector } from 'react-redux';


const ShowCounterDemo = () => {
    const counter = useSelector((state) => state.counter);
    
    return (
        <div className="App">
            <p>다른 컴포넌트에서 보여지는 카운터값 : {counter}</p>
        </div>
    );
};

export default ShowCounterDemo;