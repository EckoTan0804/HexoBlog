---
title: TypeScript Quick Tutorial
tags:
- Programming
- TypeScript
- JavaScript
- Tutorials



---

Get a quick start in TypeScript in a KISS (Keep It Simple Stupid) way!

<!--more-->

## What is TypeScript?

~~~typescript
TypeScript = JavaScript that scales
~~~

TypeScript is essentially JavaScript, but much more powerful than JavaScript.



## Useful Links

+ [TypeScript official website](https://www.typescriptlang.org/index.html)
+ [Tutorialspoint](https://www.tutorialspoint.com/typescript/index.htm)
+ [TypeScript Playground](http://www.typescriptlang.org/play/index.html)



## Prerequisites

+ Good understanding of [OOP](https://en.wikipedia.org/wiki/Object-oriented_programming) concepts
+ Basic [JavaScript](https://en.wikipedia.org/wiki/JavaScript) knowledge



## Installation

For NPM users:

```shell
$ npm install -g typescript
```

 

## Compiling

Assume that now we have a TypeScript file `HelloWorld.ts`.

At the command line, run the TypeScript compiler:

```shell
$ tsc greeter.ts
```

The result will be a file `greeter.js` which contains the same JavaScript that we fed in. 



## Basic Knowledge

### Syntax for Declaring Variable

~~~typescript
var varName : varType = varValue;
~~~



### Syntax for Declaring Function

~~~typescript
function funcName(param1: param1Type, param2: param2Type,...): returnType {
    // do sth
}
~~~



### Basic Types

**The Type System checks the validity of the supplied values, before they are stored or manipulated by the program.** This ensures that the code behaves as expected. The Type System further allows for richer code hinting and automated documentation too.

+ `Boolean`

  ~~~typescript
  var flag : boolean = false;
  ~~~

+ `Number`

+ `String`

+ `Array`

  ~~~typescript
  var users: string[] = ["A", "B", "C"];
  var list: Array<string> = ["A", "B", "C"];
  ~~~

+ `Enum`

  ~~~typescript
  enum Color {Red, Green, Blue} 
  let c: Color = Color.Green;
  ~~~

  By default, enums begin numbering their members starting at `0`. You can change this by manually setting the value of one of its members.

  ~~~typescript
  enum Color {Red = 1, Green, Blue}
  let c: Color = Color.Green;
  ~~~

  A handy feature of enums is that you can also go from a numeric value to the name of that value in the enum.

  ~~~typescript
  enum Color {Red = 1, Green, Blue}
  let colorName: string = Color[2]; // Now colorName is Green
  ~~~

+ `Any`

  The **any** data type is the super type of all types in TypeScript. It denotes a dynamic type. Using the **any** type is equivalent to opting out of type checking for a variable.

  ~~~typescript
  let notSure: any = 4;
  notSure = "maybe a string instead";
  notSure = false; // Now notSure is a boolean
  ~~~

+ `Void`

  `void` is a little like the opposite of `any`: the absence of having any type at all. 

  We can consider this as the return type of functions that do not return a value (same as in Java):

  ~~~typescript
  function greet(): void {
      console.log("Hello World!");
  }
  ~~~



  ### Class

  If you know Java or other oop programming language, this will be easy and similiar.

  ~~~typescript
  class Animal {
      
      // Attributes
      name: string;
      
      // Constructor
      constructor(name: string) {
          this.name = name;
      }
      
      // Function/Method
      sayHello():void{
          alert("hello animal:"+this.name);
      }
  }
  
  // Instantiation
  var animal = new Animal("tom");
  animal.sayHello();
  ~~~



  #### Class Inheritance

  ~~~typescript
  class Cat extends Animal {
      // override
      sayHello(): void {
          alert("hello cat:" + this.name);
      }
  }
  
  class Dog extends Animal {
      // override
      sayHello(): void {
          alert("hello dog:" + this.name);
      }
  }
  
  ~~~



  #### Modifier

  + `public`
  + `protected`
  + `private`

  + `readonly`

    must be initialized at their declaration or in the constructor.



  ### Interface

  ~~~typescript
  interface Graphic {
      width: Number;
      height: Number;
  }
  
  class Square implements Graphic {
      width: Number;
      height: Number;
  
      constructor() {
          this.width = 100;
          this.height = 100;
      }
  
      constructor(width: Number, height: Number) {
          this.height = height;
          this.width = width;
      }
  }
  ~~~



  #### Interface Inheritance

  ~~~typescript
  interface Graphic {
      width: Number;
      height: Number;
  }
  
  interface PenStroke {
      penWidth: Number;
  }
  
  interface Square extends Graphic, PenStroke {
      sideLength: number;
  }
  ~~~



  ### Modules

  + Modules are executed within their own scope, not in the global scope; 
  + Variables, functions, classes, etc. declared in a module are **not visible outside the module unless they are explicitly exported using one of the [`export`forms](https://www.typescriptlang.org/docs/handbook/modules.html#export)**. Conversely, to consume a variable, function, class, interface, etc. exported from a different module, it has to be imported using one of the [`import` forms](https://www.typescriptlang.org/docs/handbook/modules.html#import).

  + Modules are declarative; the relationships between modules are specified in terms of imports and exports at the file level.

  + In TypeScript,  any file containing a top-level `import` or `export` is considered a module. Conversely, a file **without** any top-level `import` or `export` declarations is treated as a script whose contents are available in the **global** scope (and therefore to modules as well).



  #### Export

  ~~~typescript
  module Project {
      export module Core {
          function funA() {} // Without export, not visiable outside Project
          export function funB() {
              funA();
          }
      }
  }
  
  module Project.Core {
      export function funC() {
          funA(); // Error!
          funB(); // Correct
      }
  }
  
  export {MyCore as Project.Core} // Export and Rename
  ~~~

  #### Import

  Importing is just about as easy as exporting from a module. 

  ~~~typescript
  import { ZipCodeValidator as ZCV } from "./ZipCodeValidator";
  let myValidator = new ZCV();
  ~~~
