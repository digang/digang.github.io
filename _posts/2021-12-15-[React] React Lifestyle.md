---
categories: [FrontEnd, React]
tags: [React, React_lifestyle, 리액트 라이프 스타일] # TAG 는 소문자로 작성할 것
---
## React의 LifeStyle

>리액트가 모든 요소들을 View 로 갖기 때문에 이 View를 데이터의 단방향으로 보장을 하면서 클래스 내애서 단일적으로 관리하기 위한 로직을 마련하는 기능을 일컫는다.
>리액트는 모든것이 View 로만 작동한다. 즉 Render를 기점으로 클래스를 작성해주면 된다.

```jsx
class LifeCycle extends React.Component{
  //this.state 는 어디서 사용이 가능한가?
  // render 함수, unmount 함수에서 불가능하다. 
  // 랜더링 과정중 setState 함수를 호출하는것은 말이 되지 않는다.
  // unmount도 마찬가지로 삭제하는 과정에서 setState를 하는 의미가 없다.
  constructor(props){
    super(props)

  }

  componentWillMount(){ // 컴포넌트가 랜더링 되어지기 전 !
  
  }

  render(){
    return(<></>)
  }

  componentDidMount(){ // 클래스가 실행이되었을대, 랜더링 되었을때 딱 한번 실행!

  }

  componentWillUnmount(){ // 모든 로직이 완료되었을 때, 해당 컴포넌트를 삭제 or 초기화 할때.
    
  }

	// ... 이외에도 굉장히 많다. 자세한 내용은 React 홈페이지를 참조하길 바란다.
}

export default LifeCycle;
```