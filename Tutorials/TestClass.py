class Employee:
    count = 0
    def __init__(self,name,salary):
        self.name = name
        self.salary = salary
        Employee.count += 1
    def displayCount(self):
        print("Total Employee %d" % Employee.count)
    def displayemployee(self):
        print("name : ",self.name,", Salary: ",self.salary)

    def displayemployeeA(self):
        print(self.age)

emp1 = Employee("Zara",2000)
emp2 = Employee("mars",5000)
emp1.displayemployee()
emp2.displayemployee()
emp1.age = 7
print(getattr(emp1,'age'))