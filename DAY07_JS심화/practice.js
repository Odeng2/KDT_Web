//#1. this 예제
class MainComponent extends React.component {
    constructor(props) {
        super(props);
        this.state = state;
    }

    this.handleClick = this.handClick.bind(this);
}




//프로토타입 예제

// 생성자 함수 정의
function Person(name, age) {
    this.name = name;
    this.age = age;
}

// 프로토타입에 메서드 추가
Person.prototype.greet = function() {
    console.log(`Hello, my name is ${this.name} and I am ${this.age} years old.`);
};

// 새로운 객체 생성
const alice = new Person('Alice', 30);

// 프로토타입 확인
// true 출력 (alice의 프로토타입이 Person의 프로토타임과 같음. Person의 프로토타입은 object의 프로토타입곽 같음. 연이은 프로토타입의 연속을 프로톹타입 체인이라고 부름.)
console.log(Object.getPrototypeOf(alice) === Person.prototype); 

// 프로토타입 체인 확인
console.log(Object.getPrototypeOf(Person.prototype) === Object.prototype); // true 출력
console.log(Object.getPrototypeOf(Object.prototype) === null); // true 출력

// 메서드 호출
alice.greet(); // "Hello, my name is Alice and I am 30 years old." 출력
