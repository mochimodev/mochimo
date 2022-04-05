# Mochimo Contributing Guide

<sup>*You must read and agree to the [License](LICENSE.PDF) before contribution.*</sup>

## Issues

**DO NOT submit issues regarding security vulnerabilities.**<br />
Security vulnerabilities can be damaging to the Mochimo Cryptocurrency Network, so we provide generous and appropriate bounties to those who bring such vulnerabilities to our attention discreetly. If you have found what you believe to be a security vulnerability, please contact us privately at [support@mochimo.org](mailto:support@mochimo.org) or via direct message to a "@Core Contributor" in the [Mochimo Official Discord](https://discord.mochimo.org).

**Provide steps taken to reproduce your issue**<br />
This includes documenting things like the compilation process you used, your Operating System and machine resources, terminal output (a log file is beautiful), pictures (sometimes it's hard to describe with words), etc.

**Submitting a question or suggestion? Add it to the title**<br />
To help us prioritize, and allocate appropriate resources to various issues, consider prefixing your issue title appropriately:
- "[Question]: Who/What/When/Where/Why/How?"
- "[Suggestion]: Try this instead of that"

**If submitting a question, consider searching for the answer first**<br />
If you cannot find your answer in the [Mochimo Wiki](http://www.mochiwiki.com), try our Twitter [(@mochimocrypto)](https://twitter.com/mochimocrypto?lang=en), [Reddit (r/Mochimo)](https://www.reddit.com/r/mochimo/), or the [Mochimo Official Discord](https://discord.mochimo.org), which is where you'll find our most active community of Developers and Beta Testers to help out if they can.

## Pull Requests

Please review your code with the Style Guide below and don't forget to TEST! Submitting a Pull Request that differs from the Style Guide or doesn't pass tests on support platforms will only make the process longer, if not halt it alltogether.

## Basic Style Guide for Mochimo C

### Layers of Hierarchy

Each layer of hierarchy should be indented by 3 non-breaking spaces.

```
(layer0)
   (layer1)
      (layer2)
```

### Braces

Conditionals and Loops (if, else, for, while) that require an opening brace should have the opening brace on the same line as the conditional operation separated by a space.

```c
if(something) {
   statement1;
   statement2;
}
```

### Operators

Indirection (`*`), address of (`&`), and pre & post fix (`++`)(`--`) operators should always be immediately adjacent to a variable name (without spacing):

```c
   var++   /* Correct */
   *var    /* Correct */
   &var    /* Correct */
   * var   /* Incorrect */
   & var   /* Incorrect */
```

Operators of all other kinds should not touch variable names. Example:

```c
   for(i = 0; i < 100; i++) {   /* Correct   */
   for(i=0;i<100;i++) {         /* Incorrect */
```

### Comments

Don't use C99 comments in the .c files, so Trigg doesn't have a stroke.

```c
   /* This is a comment. */
   // This is an abomination
```

### Naming

Though not generally required, the Mochimo Codebase uses a variety of case types to infer the application of each name.

```c
#define TYPE 0       /* "UPPERCASE" for #define's, or... */
#define NUM_TYPES 1  /* "SNAKE_CASE" for verbosity */

byte Running;        /* "Capitalization" for a global variable */

int fix_signals()    /* "snake_case" for a function name */
{
   char *fname;      /* "lowercase" for a scope variable, or... */
   long prev_tx_id;  /* "snake_case" (lower) for verbosity */
}
```

### Declarations and Initializations

In a function definition, variables are declared and then initialized at the top.

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

}
```

Don't mix declarations and initializations. If you only have one variable of a type, you can declare and initialize it at the same time. But multiple mixed declarations and initalizations are hard on the eyes.

```c
   unsigned char var1 = 10, var2, var3 = 30; /* unpleasant */
   unsigned char var4, var5, var6;           /* pleasant */
   unsigned char var7 = 70;                  /* pleasant */
   unsigned char var8 = 80;                  /* pleasant */
```


### Type Casting

When type casting, ensure a space exists between the closing parenthesis of the type cast and the variable name. If type casting to pointer types, ensure an additional space exists between the type and the indirection operator "`*`".

```c
   int *myintp = (int *) someint;
   /* note spacing - ^  ^ */
```

### De-referencing

When using a de-reference operation on any variable, if you are doing anything at all to that variable prior to de-referencing it, including pointer arithmetic, type casting, etc, please place that manipulation inside of parenthesis prior to de-referencing, even if precedence doesn't require it.

```c
   *((uint8_t *) &varname)   /* Correct   */
   *(uint8_t*) &varname      /* Incorrect */
```

### Readability on Small Terminals & Windows

If a layer of hierarchy consists of more than 25 lines of text, use a closing comment to indicate the condition under which the layer initiated, to let the reader know what that lonely brace at the bottom of a layer indicates.

```c
for(something) {
   ... [ > 25 lines of code ]
} /* end for(something)...*/
```

If the visual length of any line (including indentation) exceeds 76 characters and it is impractical to reduce the length of a statement, wrap the statement to the next line and apply a single additional level of hierarchy per statement.

```c
#define REALLY_LONG_PRE_PROCESSOR_STR_LITERAL \
   "Really long pre-processor string literal. " \
   "Impractically long but sometimes necessary."

int really_long_function_and_parameters(
   int parameter1, int parameter2, int parameter3)
{
   char *really_long_variable_initialization =
      "really long variable initialization...";
}
```
