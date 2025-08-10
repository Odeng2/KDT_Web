document.addEventListener("DOMContentLoaded", function(){
    //DOM 요소 접근(=필요한 DOM 요소를 정의함)
    const header = document.getElementsByClassName("header")[0];
    const nav = document.getElementsByClassName('navigation')[0];
    const nav_ul = document.getElementsByTagName("ul")[0];
    const first_btn = document.getElementsByClassName("btn")[0];
    

    //예제 #1.

    //DOM 요소를 생성해줌 (속성)
    let newAttribute = document.createAttribute("style");
    newAttribute.value = "color:black";

    //DOM에 변경사항 적용: header에 style="color:black"이라는 스타일 코드가 적용됨.
    //<header class="header main-header" style="color:black">
    header.setAttributeNode(newAttribute);


    //예제 #2.

    //DOM 요소 생성 (자식 노드)
    let newList = document.createElement("li");
    let newContent = document.createTextNode("새로운 메뉴");

    newList.appendChild(newContent);
    nav_ul.appendChild(newList);
    
    //좋지않은사용예시
    let newContent2 = "<li>새로운 메뉴</li>";ㅁ
    av_ul.appendChild(newContent2); 


    //예제 #3.
    //이벤트 리스너
    first_btn.addEventListener("click", function () {
        alert("버튼이 클릭되었습니다.")
    })

    //add를 사용하면 항상 remove를 사용해야 함.
    //함수를 외부에 정의해두는 것이 훨씬 편리한 방법임.
    first_btn.removeEventListener("click", function () {
        alert("버튼이 클릭되었습니다.")
    })

})