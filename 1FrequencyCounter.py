def freq_counter(numbers):
    freq = {}
    for num in numbers:
        freq[num] = freq.get(num, 0) + 1
    return freq

if __name__ == "__main__":
    sample = [1, 2, 2, 3, 1, 4]
    print("Frequency Count:", freq_counter(sample))
