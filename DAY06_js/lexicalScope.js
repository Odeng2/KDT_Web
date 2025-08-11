//아래의 예제들 중, 사용하고자 하는 예제의 주석을 해제한 후 활용하기



// //예제 1

// let a = 1;

// function func1() {
// 	let a = 2;
// 	return func2();
// }

// function func2() {
// 	console.log(a);    //a값이 1로 출력됨
// }

// console.log(a);    //a값이 1로 출력됨. (Dynamic Scope의 경우 a값은 2로 출력)
// func1();



// //비교 에제 2-1
// const a = 1;
// const b = 1;
// const c = 1;

// function funcA(){
// 	const b = 2;
// 	const c = 2;
	
// 	console.log("2", a, b, c);    
// 	funcB()
// }

// function funcB() {
// 	const c = 3;
	
// 	console.log("3", c, b, c);    //출력: 3 3 1 3
// }

// console.log("1", a, b, c);    
// funcA();



// //비교 예제 2-2

// const a = 1;
// const b = 1;
// const c = 1;

// function funcA(){
// 	const b = 2;
// 	const c = 2;
	
// 	console.log("2", a, b, c);
    
// 	function funcB() {
//         const c = 3;

//         console.log ("3", c, b, c);    //출력: 3 3 2 3
//     }

//     funcB();
// }


// console.log("1", a, b, c);
// funcA();
