def min_max_normalize(numbers):
 #Returns a list of numbers normalized between 0 and 1 using min-max scaling.
    min_val = min(numbers)
    max_val = max(numbers)
    if max_val == min_val:
        return [0 for _ in numbers]  # avoid division by zero
    return [(x - min_val) / (max_val - min_val) for x in numbers]

# Example 
if __name__ == "__main__":
    sample = [10, 20, 30, 40]
    print("Normalized:", min_max_normalize(sample))
