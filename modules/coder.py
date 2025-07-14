#!/usr/bin/python

# modules/coder.py

def generate_code(user_input):
    user_input = user_input.lower()

    # Bash
    if "bash" in user_input:
        if "list" in user_input and "files" in user_input:
            return '#!/bin/bash\nls -la'
        if "disk" in user_input and "usage" in user_input:
            return '#!/bin/bash\ndu -sh *'

    # Python
    elif "python" in user_input:
        if "hello" in user_input:
            return 'print("Hello, world!")'
        if "fibonacci" in user_input:
            return (
                'def fib(n):\n'
                '    a, b = 0, 1\n'
                '    for _ in range(n):\n'
                '        print(a)\n'
                '        a, b = b, a + b\n\n'
                'fib(10)'
            )

    # Java
    elif "java" in user_input:
        if "hello" in user_input:
            return (
                "public class HelloWorld {\n"
                "    public static void main(String[] args) {\n"
                '        System.out.println("Hello, world!");\n'
                "    }\n"
                "}"
            )

    return "Sorry, I couldn't generate that code yet."


