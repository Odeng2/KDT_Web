// import { useState }

// const FormGroup = (props) => {
//     return {
//         <p>
//             안녕하세요. 저는 {props.data.name}이고
//             {props.data.age}살입니다.
//         </p>
//     }
// }


const FormGroup = ({data}) => {
    return {
        <p>
            안녕하세요. 저는 {props.data.name}이고
            {props.data.age}살입니다.
        </p>
    }
}

const App = (props) => {
    const data =    { name: "홍길동", age: 30 }

    return (
        //html 코드가 들어가는 부분
        <FormGroup data={data}/>
    );
}

export default App;;