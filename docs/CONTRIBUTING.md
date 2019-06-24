# Contributing to Mochimo
<sup>*You must read and agree to the [LICENSE.PDF](LICENSE.PDF) file prior to running our code.*</sup>  
Thank you for your interest in contributing to the Mochimo Main Net repository!  
We're excited to see what the community does to further the development of Mochimo. Please take some time to go over the guidelines set out in this document to ensure your contribution process is as smooth as possible.

## Submitting Issues
**DO NOT submit issues regarding security vulnerabilities.**  
Publicly exposed security vulnerabilities can be damaging to the Mochimo Cryptocurrency Network, so we provide generous and appropriate bounties for those who bring such vulnerabilities to our attention discretely. If you have found what you believe to be a security vulnerability, please contact us privately via email (support@mochimo.org) or via Discord DM (@Core Contributor).

**DO NOT submit questions as GitHub issues**  
Submitting a question as an issue is basically "noise" in a constructive development workspace.  
There are other, more appropriate methods of getting answers to your questions. If you cannot find your answer in the [Mochimo Wiki](http://www.mochiwiki.com), try our Twitter (@mochimocrypto), Reddit (r/Mochimo), or our Official Discord which is where you'll find our most active community of Developers and Beta Testers to help out if they can.

**Provide steps taken to reproduce your issue**  
This includes things like the compilation process you used, your Operating System and machine resources, Terminal output (a log file is beautiful), Pictures (pictures are so good), etc.

**If submitting a suggestion, make a note of it in the issue title**  
This helps us prioritize and assign issues appropriately. A simple "SUGGESTION: This is my suggestion" will do fine.

## Submitting Pull Requests
Please review your code with the Style Guide below and TEST your Pull Requests before you send them. Evidence of a WORKING Pull Request in the form of pictures or terminal output help us understand the improvements/changes you have made and may also speed up the review process.

## Basic Style Guide for Mochimo C
#### Type Casting
Please insert a space between the closing parenthesis of the type cast and the variable. The indirection operator (`*`) should have a space between it and the type:
```
(type *) varname
```

#### Braces and Spacing
Conditionals and Loops (if, else, for, while) that require an opening brace should have the opening brace on the same line as the conditional operation separated by a space. Each layer of the hierarchy should be indented by 3 spaces:
```c
if(something) {
   statement1;
   statement2;
}
```
For long loops, conditional statements, functions, etc - if the block takes up more than 25 lines of text, please have a closing comment:
```c
for(something) {
   [Line 1 - 25]
} /* end for(something)...*/
```
In other words if you can't see the top of the loop on one page, then let people know what that lonely brace at the bottom is actually closing out.

Indirection (`*`) and address of (`&`) operators should always be immediately adjacent to a variable name:  
```c
   *var	   /* Correct */
   &var	   /* Correct */
   * var   /* Incorrect */
   & var   /* Incorrect */
```

Operators of all other kinds should not touch variable names. Example:  
```c
   for(i = 0; i < 100; i++) {   /* Correct   */
   for(i=0;i<100;i++) {         /* Incorrect */
```

#### Comments
Please don't use C99 comments in the .c files, so that Trigg doesn't have a stroke.  So:
```c
   /* This is a comment. */
   // This is an abomination
```

#### De-referencing
When using a de-reference operation on any variable, if you are doing anything at all to that variable prior to de-referencing it, including pointer arithmetic, type casting, etc, please place that manipulation inside of parenthesis prior to de-referencing, even if precedence doesn't require it. Example:  
```c
   *((uint8_t *) &varname)   /* Correct   */
   *(uint8_t*) &varname      /* Incorrect */
```

#### Globals
In the mochimo code base (c only, not Cuda), a variable is a GLOBAL and defined in data.c if the first letter is Capitalized.
```c
byte Running;   /* GLOBAL */
```

#### Declarations and Initializations
In a function definition, variables are declared, and then initialized at the top:
```c
void my_function()
{
   /* Declarations Here */
   uint8_t var1, var2, var3;
   uint16_t varname = 10;

   /* Initialization Here */
   var1 = 10;
   var2 = 20;
   
   ...
   
} /* end my_func... */
```
Please don't mix declarations and initializations. If you only have one variable of a type, you can declare and initialize it at the same time. But something like this is just hard on the eyes:  
```c
   uint8_t var1 = 10, var2, var3 = 30;   /* Unpleasant */
```