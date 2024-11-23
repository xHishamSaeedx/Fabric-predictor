import pandas as pd

def split_csv():
    # Read the original CSV
    df = pd.read_csv('fabric.csv')
    
    # Create a new list to store the expanded rows
    new_rows = []
    
    # Words to exclude
    exclude_words = {'and', 'or', 'the', 'a', 'an', 'wear', 'clothing'}
    
    # Iterate through each row in the original dataframe
    for _, row in df.iterrows():
        # Split the Best Use into individual uses and clean them
        best_uses = row['Best Use'].split()
        # Filter out connecting words and basic words
        best_uses = [use for use in best_uses if use.lower() not in exclude_words]
        
        # Combine multi-word uses
        i = 0
        while i < len(best_uses) - 1:
            # If current word is typically part of a compound (e.g., "winter" followed by "coats")
            if best_uses[i].lower() in {'winter', 'summer', 'evening', 'casual', 'eco'} and i < len(best_uses) - 1:
                best_uses[i] = f"{best_uses[i]}-{best_uses[i+1]}"
                best_uses.pop(i+1)
            else:
                i += 1
        
        # Create a new row for each best use
        for use in best_uses:
            new_row = row.copy()
            new_row['Best Use'] = use.strip()
            new_rows.append(new_row)
    
    # Create new dataframe from the expanded rows
    new_df = pd.DataFrame(new_rows)
    
    # Remove any rows where Best Use is empty or just connecting words
    new_df = new_df[new_df['Best Use'].str.len() > 1]
    
    # Save the restructured CSV
    new_df.to_csv('fabric_restructured.csv', index=False)
    print("CSV has been restructured successfully!")

if __name__ == "__main__":
    split_csv() 