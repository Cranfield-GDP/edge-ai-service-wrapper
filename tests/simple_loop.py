def process_data(data):
    """
    Process data with multiple conditions and loops.

    Args:
    data (list): List of data to process.

    Returns:
    list: Processed data.
    """
    processed_data = []
    for item in data:
        if isinstance(item, int):
            while item > 0:
                processed_data.append(item)
                item -= 1
        elif isinstance(item, float):
            while item > 0:
                processed_data.append(item)
                item -= 1
        else:
            processed_data.append(item)
    return processed_data


# Example usage:
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
processed_data = process_data(data)
print(processed_data)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
