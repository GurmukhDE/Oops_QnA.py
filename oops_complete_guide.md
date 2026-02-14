# Complete OOP Guide for Data Engineering

## 📚 PART 1: FULL REVISION NOTES (One-Page Summary)

### The 4 Pillars of OOP

**1. ENCAPSULATION** - Bundling data and methods; hiding internal details
- Data hiding using private/protected/public access modifiers
- Getters/Setters for controlled access
- Example: `__init__()` for initialization, `_private` and `__very_private` attributes

**2. INHERITANCE** - Child class acquires properties/methods from parent
- Types: Single, Multiple, Multilevel, Hierarchical, Hybrid
- `super()` to call parent class methods
- Method Resolution Order (MRO) - order Python searches for methods

**3. POLYMORPHISM** - Same interface, different implementations
- Method Overriding: Child redefines parent method
- Method Overloading: Same name, different parameters (using default args in Python)
- Duck Typing: "If it walks like a duck and quacks like a duck..."

**4. ABSTRACTION** - Hiding complexity, showing only essentials
- Abstract Base Classes (ABC) with `@abstractmethod`
- Interfaces (protocols in Python)
- Cannot instantiate abstract classes

### Key Concepts
- **Class**: Blueprint for objects
- **Object**: Instance of a class
- **Constructor**: `__init__()` method
- **Destructor**: `__del__()` method
- **self**: Reference to current instance
- **cls**: Reference to class itself (in classmethods)
- **@staticmethod**: No self/cls, utility function
- **@classmethod**: Takes cls, can modify class state
- **@property**: Method as attribute (getter)
- **Composition**: "Has-a" relationship
- **Aggregation**: Weak "has-a" (parts can exist independently)

### Special Methods (Dunder/Magic Methods)
```
__init__, __str__, __repr__, __len__, __getitem__, __setitem__
__add__, __eq__, __lt__, __enter__, __exit__
```

---

## 💻 PART 2: 10 CODING PRACTICE QUESTIONS

### Q1: Design a Library Management System
**Requirements**: Book, Member, Library classes with borrow/return functionality

```python
class Book:
    def __init__(self, book_id, title, author):
        self.book_id = book_id
        self.title = title
        self.author = author
        self.is_available = True
    
    def __str__(self):
        return f"{self.title} by {self.author}"

class Member:
    def __init__(self, member_id, name):
        self.member_id = member_id
        self.name = name
        self.borrowed_books = []
    
    def borrow_book(self, book):
        if book.is_available:
            self.borrowed_books.append(book)
            book.is_available = False
            return True
        return False
    
    def return_book(self, book):
        if book in self.borrowed_books:
            self.borrowed_books.remove(book)
            book.is_available = True
            return True
        return False

class Library:
    def __init__(self):
        self.books = []
        self.members = []
    
    def add_book(self, book):
        self.books.append(book)
    
    def register_member(self, member):
        self.members.append(member)
    
    def find_available_books(self):
        return [book for book in self.books if book.is_available]
```

### Q2: Implement Shape Hierarchy (Polymorphism)
**Task**: Create abstract Shape class with Circle, Rectangle, Triangle

```python
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
    def area(self):
        return self.length * self.width
    
    def perimeter(self):
        return 2 * (self.length + self.width)

class Triangle(Shape):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def area(self):
        s = (self.a + self.b + self.c) / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))
    
    def perimeter(self):
        return self.a + self.b + self.c

# Usage
shapes = [Circle(5), Rectangle(4, 6), Triangle(3, 4, 5)]
for shape in shapes:
    print(f"Area: {shape.area():.2f}, Perimeter: {shape.perimeter():.2f}")
```

### Q3: Bank Account with Inheritance
**Task**: BankAccount base class, SavingsAccount and CheckingAccount derived classes

```python
class BankAccount:
    def __init__(self, account_number, balance=0):
        self._account_number = account_number
        self._balance = balance
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount
            return True
        return False
    
    def get_balance(self):
        return self._balance

class SavingsAccount(BankAccount):
    def __init__(self, account_number, balance=0, interest_rate=0.02):
        super().__init__(account_number, balance)
        self.interest_rate = interest_rate
    
    def add_interest(self):
        interest = self._balance * self.interest_rate
        self._balance += interest
        return interest

class CheckingAccount(BankAccount):
    def __init__(self, account_number, balance=0, overdraft_limit=500):
        super().__init__(account_number, balance)
        self.overdraft_limit = overdraft_limit
    
    def withdraw(self, amount):
        if 0 < amount <= (self._balance + self.overdraft_limit):
            self._balance -= amount
            return True
        return False

# Usage
savings = SavingsAccount("SA001", 1000)
savings.add_interest()
print(f"Savings Balance: {savings.get_balance()}")

checking = CheckingAccount("CA001", 500, 300)
checking.withdraw(700)  # Can go into overdraft
print(f"Checking Balance: {checking.get_balance()}")
```

### Q4: Singleton Design Pattern
**Task**: Implement a Database connection class that ensures only one instance

```python
class Database:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, host="localhost", port=5432):
        if not hasattr(self, 'initialized'):
            self.host = host
            self.port = port
            self.connection = None
            self.initialized = True
    
    def connect(self):
        if not self.connection:
            self.connection = f"Connected to {self.host}:{self.port}"
        return self.connection
    
    def disconnect(self):
        self.connection = None

# Test Singleton
db1 = Database("localhost", 5432)
db2 = Database("remote", 3306)
print(db1 is db2)  # True - same instance
print(db1.host)    # localhost (first initialization)
```

### Q5: Employee Management with Multiple Inheritance
**Task**: Employee, Developer, Manager, TechLead (multiple inheritance)

```python
class Employee:
    def __init__(self, emp_id, name, salary):
        self.emp_id = emp_id
        self.name = name
        self.salary = salary
    
    def get_details(self):
        return f"ID: {self.emp_id}, Name: {self.name}, Salary: ${self.salary}"

class Developer(Employee):
    def __init__(self, emp_id, name, salary, programming_languages):
        super().__init__(emp_id, name, salary)
        self.programming_languages = programming_languages
    
    def code_review(self):
        return f"{self.name} is reviewing code"

class Manager(Employee):
    def __init__(self, emp_id, name, salary, team_size):
        super().__init__(emp_id, name, salary)
        self.team_size = team_size
    
    def conduct_meeting(self):
        return f"{self.name} is conducting a meeting with {self.team_size} members"

class TechLead(Developer, Manager):
    def __init__(self, emp_id, name, salary, programming_languages, team_size):
        Developer.__init__(self, emp_id, name, salary, programming_languages)
        Manager.__init__(self, emp_id, name, salary, team_size)
    
    def get_role(self):
        return f"{self.name} is a Tech Lead managing {self.team_size} developers"

# Usage
tl = TechLead("TL001", "Alice", 120000, ["Python", "Java"], 5)
print(tl.get_details())
print(tl.code_review())
print(tl.conduct_meeting())
print(TechLead.__mro__)  # Method Resolution Order
```

### Q6: Custom Iterator and Iterable
**Task**: Create a custom Range class that works with for loops

```python
class CustomRange:
    def __init__(self, start, end, step=1):
        self.start = start
        self.end = end
        self.step = step
    
    def __iter__(self):
        self.current = self.start
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += self.step
        return value

# Usage
for num in CustomRange(0, 10, 2):
    print(num, end=" ")  # 0 2 4 6 8

# Alternative using generator
class BetterRange:
    def __init__(self, start, end, step=1):
        self.start = start
        self.end = end
        self.step = step
    
    def __iter__(self):
        current = self.start
        while current < self.end:
            yield current
            current += self.step
```

### Q7: Context Manager (with statement)
**Task**: Create a FileManager class for safe file handling

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
        if exc_type is not None:
            print(f"Exception occurred: {exc_value}")
        return False  # Don't suppress exceptions

# Usage
with FileManager("test.txt", "w") as f:
    f.write("Hello, OOP!")

# Alternative using contextlib
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    try:
        f = open(filename, mode)
        yield f
    finally:
        f.close()
```

### Q8: Operator Overloading
**Task**: Create a Vector class with mathematical operations

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y

# Usage
v1 = Vector(2, 3)
v2 = Vector(4, 5)
print(v1 + v2)      # Vector(6, 8)
print(v1 * 3)       # Vector(6, 9)
print(v1 == v2)     # False
print(v1.dot(v2))   # 23
```

### Q9: Property Decorators and Validation
**Task**: Create a Person class with validated properties

```python
class Person:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email
    
    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, value):
        if not isinstance(value, int):
            raise TypeError("Age must be an integer")
        if value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        self._age = value
    
    @property
    def email(self):
        return self._email
    
    @email.setter
    def email(self, value):
        if '@' not in value:
            raise ValueError("Invalid email format")
        self._email = value
    
    @property
    def is_adult(self):
        return self._age >= 18
    
    def __str__(self):
        return f"{self.name}, {self.age} years old ({self.email})"

# Usage
person = Person("John", 25, "john@example.com")
print(person.is_adult)  # True
# person.age = -5  # Raises ValueError
# person.email = "invalid"  # Raises ValueError
```

### Q10: Composite Design Pattern (Part-Whole Hierarchy)
**Task**: File System with files and directories

```python
from abc import ABC, abstractmethod

class FileSystemComponent(ABC):
    @abstractmethod
    def show_details(self, indent=0):
        pass
    
    @abstractmethod
    def get_size(self):
        pass

class File(FileSystemComponent):
    def __init__(self, name, size):
        self.name = name
        self.size = size
    
    def show_details(self, indent=0):
        print("  " * indent + f"File: {self.name} ({self.size} KB)")
    
    def get_size(self):
        return self.size

class Directory(FileSystemComponent):
    def __init__(self, name):
        self.name = name
        self.children = []
    
    def add(self, component):
        self.children.append(component)
    
    def remove(self, component):
        self.children.remove(component)
    
    def show_details(self, indent=0):
        print("  " * indent + f"Directory: {self.name}")
        for child in self.children:
            child.show_details(indent + 1)
    
    def get_size(self):
        return sum(child.get_size() for child in self.children)

# Usage
root = Directory("root")
home = Directory("home")
user = Directory("user")

file1 = File("document.txt", 100)
file2 = File("image.jpg", 500)
file3 = File("code.py", 50)

user.add(file1)
user.add(file2)
home.add(user)
home.add(file3)
root.add(home)

root.show_details()
print(f"Total Size: {root.get_size()} KB")
```

---

## 🎯 PART 3: TRICKY INTERVIEW QUESTIONS

### Q1: What happens when you inherit from multiple classes with the same method name?
**Answer**: Python uses Method Resolution Order (MRO) following C3 linearization algorithm. Check with `ClassName.__mro__` or `ClassName.mro()`. The leftmost class in inheritance takes priority.

```python
class A:
    def method(self):
        return "A"

class B:
    def method(self):
        return "B"

class C(A, B):
    pass

print(C().method())  # "A" - leftmost parent
print(C.__mro__)     # C -> A -> B -> object
```

### Q2: What's the difference between `__str__` and `__repr__`?
**Answer**: 
- `__str__`: Human-readable string, for end users, used by `str()` and `print()`
- `__repr__`: Unambiguous representation, for developers, used by `repr()` and in interactive mode
- Best practice: `__repr__` should return a string that could recreate the object

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Point at ({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
```

### Q3: Explain the difference between `@staticmethod`, `@classmethod`, and instance methods
**Answer**:
- **Instance method**: Takes `self`, operates on instance
- **@classmethod**: Takes `cls`, can modify class state, used for factory methods
- **@staticmethod**: No self/cls, can't modify instance or class, utility function

```python
class MyClass:
    class_variable = 0
    
    def instance_method(self):
        return f"Instance method, {self}"
    
    @classmethod
    def class_method(cls):
        cls.class_variable += 1
        return f"Class method, {cls}"
    
    @staticmethod
    def static_method():
        return "Static method, no self or cls"
```

### Q4: What is the diamond problem and how does Python solve it?
**Answer**: Diamond problem occurs in multiple inheritance when a class inherits from two classes that inherit from a common base. Python solves this using MRO (C3 linearization) ensuring each class appears only once in the hierarchy.

```python
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")
        super().method()

class C(A):
    def method(self):
        print("C")
        super().method()

class D(B, C):
    def method(self):
        print("D")
        super().method()

D().method()  # D, B, C, A (MRO order)
print(D.__mro__)
```

### Q5: What's the difference between composition and inheritance?
**Answer**:
- **Inheritance (is-a)**: Child "is a" type of Parent
- **Composition (has-a)**: Object "has a" component
- Favor composition over inheritance for flexibility

```python
# Inheritance
class Animal:
    def move(self):
        pass

class Dog(Animal):  # Dog IS-A Animal
    def move(self):
        return "Running"

# Composition
class Engine:
    def start(self):
        return "Engine started"

class Car:  # Car HAS-A Engine
    def __init__(self):
        self.engine = Engine()
    
    def start(self):
        return self.engine.start()
```

### Q6: Can you modify a private variable from outside the class?
**Answer**: Yes! Python's "privacy" is by convention, not enforcement. Single underscore `_var` is a convention. Double underscore `__var` uses name mangling (`_ClassName__var`) but is still accessible.

```python
class MyClass:
    def __init__(self):
        self._protected = "Protected"
        self.__private = "Private"

obj = MyClass()
print(obj._protected)  # Works
# print(obj.__private)  # AttributeError
print(obj._MyClass__private)  # Works! Name mangling revealed
```

### Q7: What's the difference between shallow copy and deep copy in OOP?
**Answer**:
- **Shallow copy**: Copies object but not nested objects (references remain)
- **Deep copy**: Recursively copies all nested objects

```python
import copy

class Inner:
    def __init__(self, value):
        self.value = value

class Outer:
    def __init__(self, inner):
        self.inner = inner

original = Outer(Inner(10))
shallow = copy.copy(original)
deep = copy.deepcopy(original)

original.inner.value = 20
print(shallow.inner.value)  # 20 (shared reference)
print(deep.inner.value)     # 10 (independent copy)
```

### Q8: How do you prevent a class from being inherited?
**Answer**: Override `__init_subclass__` or use a metaclass. Python doesn't have a final keyword like Java.

```python
class FinalClass:
    def __init_subclass__(cls):
        raise TypeError(f"Cannot inherit from FinalClass")

# This will raise TypeError
# class SubClass(FinalClass):
#     pass
```

### Q9: What is monkey patching and when would you use it?
**Answer**: Dynamically modifying a class or module at runtime. Useful for testing, adding functionality, or fixing bugs in third-party libraries without modifying source.

```python
class Calculator:
    def add(self, a, b):
        return a + b

# Monkey patching
def new_add(self, a, b):
    print("Addition called")
    return a + b

Calculator.add = new_add

calc = Calculator()
calc.add(2, 3)  # Prints "Addition called" then returns 5
```

### Q10: Explain the concept of "duck typing"
**Answer**: "If it walks like a duck and quacks like a duck, it's a duck." Python doesn't check types explicitly; it checks if objects have required methods/attributes.

```python
class Duck:
    def quack(self):
        return "Quack!"

class Person:
    def quack(self):
        return "I'm imitating a duck!"

def make_it_quack(duck):
    # Doesn't care about type, just that it has quack()
    return duck.quack()

print(make_it_quack(Duck()))    # Works
print(make_it_quack(Person()))  # Also works!
```

### Q11: What's the difference between `is` and `==`?
**Answer**:
- `is`: Identity comparison (same object in memory)
- `==`: Equality comparison (same value, calls `__eq__`)

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)  # True (same value)
print(a is b)  # False (different objects)
print(a is c)  # True (same object)
```

### Q12: How do you make a class callable?
**Answer**: Implement `__call__` method

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)
print(double(5))  # 10 (object used like a function)
```

---

## 🏗️ PART 4: REAL DATA ENGINEERING OOP DESIGN EXAMPLE

### ETL Pipeline Framework with OOP

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= ABSTRACT BASE CLASSES =============

class DataSource(ABC):
    """Abstract base class for all data sources"""
    
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Extract data from source"""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to data source"""
        pass

class Transformer(ABC):
    """Abstract base class for all transformers"""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        pass

class DataSink(ABC):
    """Abstract base class for all data sinks"""
    
    @abstractmethod
    def load(self, data: pd.DataFrame) -> bool:
        """Load data to destination"""
        pass

# ============= CONCRETE DATA SOURCES =============

class CSVSource(DataSource):
    """Extract data from CSV files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def validate_connection(self) -> bool:
        try:
            import os
            return os.path.exists(self.file_path)
        except Exception as e:
            logger.error(f"CSV validation failed: {e}")
            return False
    
    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting data from CSV: {self.file_path}")
        return pd.read_csv(self.file_path)

class DatabaseSource(DataSource):
    """Extract data from database"""
    
    def __init__(self, connection_string: str, query: str):
        self.connection_string = connection_string
        self.query = query
        self.connection = None
    
    def validate_connection(self) -> bool:
        # Simulate connection validation
        logger.info("Validating database connection...")
        return True
    
    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting data from database")
        # Simulated: pd.read_sql(self.query, self.connection)
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100, 200, 300]
        })

class APISource(DataSource):
    """Extract data from REST API"""
    
    def __init__(self, endpoint: str, headers: Dict = None):
        self.endpoint = endpoint
        self.headers = headers or {}
    
    def validate_connection(self) -> bool:
        logger.info(f"Validating API endpoint: {self.endpoint}")
        return True
    
    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting data from API: {self.endpoint}")
        # Simulated API call
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'metric': ['sales'],
            'value': [5000]
        })

# ============= CONCRETE TRANSFORMERS =============

class CleaningTransformer(Transformer):
    """Remove null values and duplicates"""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data: removing nulls and duplicates")
        return data.dropna().drop_duplicates()

class AggregationTransformer(Transformer):
    """Aggregate data by specified columns"""
    
    def __init__(self, group_by: List[str], agg_func: Dict[str, str]):
        self.group_by = group_by
        self.agg_func = agg_func
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Aggregating by {self.group_by}")
        return data.groupby(self.group_by).agg(self.agg_func).reset_index()

class EnrichmentTransformer(Transformer):
    """Enrich data with additional columns"""
    
    def __init__(self, enrichment_logic: callable):
        self.enrichment_logic = enrichment_logic
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Enriching data with additional columns")
        return self.enrichment_logic(data)

# ============= CONCRETE DATA SINKS =============

class CSVSink(DataSink):
    """Load data to CSV file"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
    
    def load(self, data: pd.DataFrame) -> bool:
        try:
            logger.info(f"Loading data to CSV: {self.output_path}")
            data.to_csv(self.output_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Failed to load to CSV: {e}")
            return False

class DatabaseSink(DataSink):
    """Load data to database"""
    
    def __init__(self, connection_string: str, table_name: str):
        self.connection_string = connection_string
        self.table_name = table_name
    
    def load(self, data: pd.DataFrame) -> bool:
        try:
            logger.info(f"Loading data to database table: {self.table_name}")
            # Simulated: data.to_sql(self.table_name, self.connection)
            return True
        except Exception as e:
            logger.error(f"Failed to load to database: {e}")
            return False

class DataWarehouseSink(DataSink):
    """Load data to data warehouse (e.g., Snowflake, BigQuery)"""
    
    def __init__(self, warehouse_config: Dict):
        self.warehouse_config = warehouse_config
    
    def load(self, data: pd.DataFrame) -> bool:
        logger.info(f"Loading data to data warehouse")
        # Simulated warehouse load
        return True

# ============= ETL PIPELINE ORCHESTRATOR =============

class ETLPipeline:
    """Main ETL Pipeline orchestrator using composition"""
    
    def __init__(self, name: str):
        self.name = name
        self.sources: List[DataSource] = []
        self.transformers: List[Transformer] = []
        self.sinks: List[DataSink] = []
        self.metadata = {
            'created_at': datetime.now(),
            'runs': 0,
            'last_run': None,
            'status': 'initialized'
        }
    
    def add_source(self, source: DataSource):
        """Add a data source to the pipeline"""
        self.sources.append(source)
        return self
    
    def add_transformer(self, transformer: Transformer):
        """Add a transformer to the pipeline"""
        self.transformers.append(transformer)
        return self
    
    def add_sink(self, sink: DataSink):
        """Add a data sink to the pipeline"""
        self.sinks.append(sink)
        return self
    
    def run(self) -> bool:
        """Execute the ETL pipeline"""
        logger.info(f"Starting ETL Pipeline: {self.name}")
        
        try:
            # Extract from all sources
            dataframes = []
            for source in self.sources:
                if source.validate_connection():
                    df = source.extract()
                    dataframes.append(df)
                else:
                    raise ConnectionError(f"Failed to connect to {source}")
            
            # Combine all extracted data
            combined_data = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Extracted {len(combined_data)} rows")
            
            # Apply all transformations
            transformed_data = combined_data
            for transformer in self.transformers:
                transformed_data = transformer.transform(transformed_data)
            
            logger.info(f"Transformed to {len(transformed_data)} rows")
            
            # Load to all sinks
            for sink in self.sinks:
                sink.load(transformed_data)
            
            # Update metadata
            self.metadata['runs'] += 1
            self.metadata['last_run'] = datetime.now()
            self.metadata['status'] = 'success'
            
            logger.info(f"Pipeline {self.name} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.metadata['status'] = 'failed'
            return False
    
    def get_metadata(self) -> Dict:
        """Get pipeline metadata"""
        return self.metadata

# ============= PIPELINE FACTORY (Factory Pattern) =============

class PipelineFactory:
    """Factory for creating predefined pipeline configurations"""
    
    @staticmethod
    def create_sales_pipeline() -> ETLPipeline:
        """Create a sales data processing pipeline"""
        pipeline = ETLPipeline("Sales Analytics Pipeline")
        
        # Add sources
        pipeline.add_source(CSVSource("sales_data.csv"))
        pipeline.add_source(APISource("https://api.example.com/sales"))
        
        # Add transformers
        pipeline.add_transformer(CleaningTransformer())
        pipeline.add_transformer(
            AggregationTransformer(
                group_by=['product_id'],
                agg_func={'revenue': 'sum', 'quantity': 'sum'}
            )
        )
        
        # Add sinks
        pipeline.add_sink(DatabaseSink("postgresql://...", "sales_summary"))
        pipeline.add_sink(CSVSink("output/sales_report.csv"))
        
        return pipeline
    
    @staticmethod
    def create_customer_pipeline() -> ETLPipeline:
        """Create a customer data processing pipeline"""
        pipeline = ETLPipeline("Customer Data Pipeline")
        
        # Custom enrichment logic
        def enrich_customer_data(df):
            df['customer_tier'] = df['total_purchases'].apply(
                lambda x: 'Gold' if x > 10000 else 'Silver' if x > 5000 else 'Bronze'
            )
            return df
        
        pipeline.add_source(DatabaseSource("connection_string", "SELECT * FROM customers"))
        pipeline.add_transformer(CleaningTransformer())
        pipeline.add_transformer(EnrichmentTransformer(enrich_customer_data))
        pipeline.add_sink(DataWarehouseSink({'warehouse': 'snowflake'}))
        
        return pipeline

# ============= USAGE EXAMPLE =============

def main():
    """Demonstrate the ETL pipeline framework"""
    
    # Create pipeline using factory
    sales_pipeline = PipelineFactory.create_sales_pipeline()
    
    # Run the pipeline
    success = sales_pipeline.run()
    
    # Check metadata
    metadata = sales_pipeline.get_metadata()
    print(f"\nPipeline Metadata:")
    print(f"Status: {metadata['status']}")
    print(f"Total Runs: {metadata['runs']}")
    print(f"Last Run: {metadata['last_run']}")
    
    # Create custom pipeline (builder pattern)
    custom_pipeline = (ETLPipeline("Custom Pipeline")
                      .add_source(CSVSource("input.csv"))
                      .add_transformer(CleaningTransformer())
                      .add_sink(CSVSink("output.csv")))
    
    custom_pipeline.run()

if __name__ == "__main__":
    main()

# ============= KEY OOP PRINCIPLES DEMONSTRATED =============
"""
1. ABSTRACTION: Abstract base classes (DataSource, Transformer, DataSink)
2. INHERITANCE: Concrete implementations inherit from abstract bases
3. ENCAPSULATION: Private data, controlled access through methods
4. POLYMORPHISM: Different sources/transformers/sinks interchangeable
5. COMPOSITION: ETLPipeline composes multiple objects
6. FACTORY PATTERN: PipelineFactory creates configured pipelines
7. BUILDER PATTERN: Method chaining for pipeline construction
8. SINGLE RESPONSIBILITY: Each class has one clear purpose
9. OPEN/CLOSED: Open for extension (new sources), closed for modification
10. DEPENDENCY INJECTION: Pipeline receives dependencies, not creates them
"""
```

### Why This Design Is Powerful:

1. **Extensibility**: Add new sources/transformers/sinks without modifying existing code
2. **Testability**: Each component can be tested independently
3. **Reusability**: Components can be mixed and matched for different pipelines
4. **Maintainability**: Clear structure, easy to understand and modify
5. **Scalability**: Can add parallel processing, error handling, monitoring easily

---

## 📝 PART 5: 20 TRICKY MCQs

**Q1**: What will be the output?
```python
class A:
    x = 10

a = A()
b = A()
a.x = 20
print(b.x)
```
a) 10  ✓
b) 20
c) Error
d) None

**Answer: a)** Instance variable `a.x` is created, but `b.x` still refers to class variable.

---

**Q2**: Which is TRUE about `__init__`?
a) It's a constructor
b) It initializes the object
c) It's called automatically ✓
d) All of the above ✓

**Answer: d)** All statements are correct.

---

**Q3**: What's the output?
```python
class Parent:
    def __init__(self):
        self.value = "Parent"

class Child(Parent):
    def __init__(self):
        self.value = "Child"

c = Child()
print(c.value)
```
a) Parent
b) Child ✓
c) Error
d) None

**Answer: b)** Child's `__init__` overrides Parent's. No `super().__init__()` called.

---

**Q4**: Which method makes an object callable?
a) `__init__`
b) `__call__` ✓
c) `__str__`
d) `__repr__`

**Answer: b)** `__call__` allows using object like a function.

---

**Q5**: What's the output?
```python
class Test:
    def __init__(self):
        self.__x = 10
    
t = Test()
print(t._Test__x)
```
a) Error
b) 10 ✓
c) None
d) AttributeError

**Answer: b)** Name mangling makes `__x` accessible as `_Test__x`.

---

**Q6**: Which is NOT a type of inheritance?
a) Single
b) Multiple
c) Circular ✓
d) Multilevel

**Answer: c)** Circular inheritance is not valid/supported.

---

**Q7**: What does `super()` do?
a) Calls parent class method ✓
b) Creates new object
c) Deletes object
d) None

**Answer: a)** `super()` gives access to parent class methods.

---

**Q8**: What's the output?
```python
class A:
    count = 0
    def __init__(self):
        A.count += 1

a1 = A()
a2 = A()
print(A.count)
```
a) 0
b) 1
c) 2 ✓
d) Error

**Answer: c)** Class variable shared across instances.

---

**Q9**: Which is TRUE about abstract classes?
a) Can have abstract methods ✓
b) Cannot be instantiated ✓
c) Must inherit from ABC ✓
d) All of the above ✓

**Answer: d)** All statements are correct.

---

**Q10**: What's the difference between `__str__` and `__repr__`?
a) `__str__` for users, `__repr__` for developers ✓
b) No difference
c) `__str__` is faster
d) `__repr__` is deprecated

**Answer: a)** `__str__` is human-readable, `__repr__` is unambiguous.

---

**Q11**: What's the output?
```python
class MyClass:
    @staticmethod
    def method():
        return "Static"

print(MyClass.method())
```
a) Error
b) Static ✓
c) None
d) MyClass

**Answer: b)** Static methods can be called without instantiation.

---

**Q12**: Which decorator makes a method a class method?
a) `@staticmethod`
b) `@classmethod` ✓
c) `@property`
d) `@abstractmethod`

**Answer: b)** `@classmethod` takes `cls` as first parameter.

---

**Q13**: What's the output?
```python
class A:
    def __init__(self):
        self.x = 1

class B(A):
    def __init__(self):
        super().__init__()
        self.x = 2

b = B()
print(b.x)
```
a) 1
b) 2 ✓
c) Error
d) None

**Answer: b)** Child's assignment happens after parent's.

---

**Q14**: Which is TRUE about encapsulation?
a) Bundles data and methods ✓
b) Hides internal details ✓
c) Uses access modifiers ✓
d) All of the above ✓

**Answer: d)** Encapsulation encompasses all these concepts.

---

**Q15**: What makes Python's inheritance different from Java?
a) Python supports multiple inheritance ✓
b) Python has no interfaces (uses ABC)
c) Python is dynamically typed
d) All of the above ✓

**Answer: d)** All are key differences.

---

**Q16**: What's the output?
```python
class Test:
    def __init__(self, x):
        self.x = x
    
    def __eq__(self, other):
        return self.x == other.x

t1 = Test(5)
t2 = Test(5)
print(t1 == t2)
```
a) False
b) True ✓
c) Error
d) None

**Answer: b)** Custom `__eq__` compares `x` values.

---

**Q17**: Which is NOT a magic/dunder method?
a) `__init__`
b) `__main__` ✓
c) `__str__`
d) `__len__`

**Answer: b)** `__main__` is not a magic method.

---

**Q18**: What's the output?
```python
class A:
    x = []
    
a = A()
b = A()
a.x.append(1)
print(len(b.x))
```
a) 0
b) 1 ✓
c) Error
d) 2

**Answer: b)** Mutable class variable is shared!

---

**Q19**: Which principle states "favor composition over inheritance"?
a) Single Responsibility
b) Open/Closed
c) Liskov Substitution
d) Design principle (not SOLID) ✓

**Answer: d)** General design principle, not part of SOLID.

---

**Q20**: What's the output?
```python
class Parent:
    def show(self):
        print("Parent")

class Child(Parent):
    def show(self):
        print("Child")
        super().show()

Child().show()
```
a) Parent
b) Child
c) Child\nParent ✓
d) Error

**Answer: c)** Prints "Child" then calls parent's show().

---

## 🎓 BONUS: Quick Reference

### Common Design Patterns in OOP

1. **Singleton**: Only one instance (Database connection)
2. **Factory**: Creates objects without specifying exact class
3. **Builder**: Constructs complex objects step by step
4. **Observer**: Notifies dependents of state changes
5. **Strategy**: Select algorithm at runtime
6. **Decorator**: Add behavior without modifying class
7. **Adapter**: Make incompatible interfaces work together
8. **Composite**: Tree structures (files/folders)

### SOLID Principles

- **S**ingle Responsibility: One class, one job
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable
- **I**nterface Segregation: Many specific interfaces > one general
- **D**ependency Inversion: Depend on abstractions, not concretions

### Common Mistakes to Avoid

1. Using mutable default arguments (`def __init__(self, items=[])`)
2. Not using `super().__init__()` in inheritance
3. Modifying class variables when you meant instance variables
4. Overusing inheritance instead of composition
5. Not implementing `__repr__` for debugging
6. Catching too broad exceptions in methods
7. Not validating input in property setters
8. Creating circular dependencies between classes

---

**Good luck with your OOP mastery! 🚀**
