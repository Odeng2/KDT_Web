// ./components/example/childComponent.js

const ChildComponent = (props) => {

    const handleChange = (e) => {
        props.onChange(e.target.value);
    }

    return (
        <div>
            <h1>자식의 상태값 : {props.dataForChild}</h1>
            <input
                type="text"
                onChange={handleChange}
            />
        </div>
    )
};

export default ChildComponent;