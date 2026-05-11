import pandas as pd

df = pd.read_csv('submission.csv')

# Shenzhen specific logic (ends in _0 or _1)
sz_df = df[df['image_id'].str.startswith('CHNCXR_')]

if len(sz_df) > 0:
    sz_df['true_label'] = sz_df['image_id'].apply(lambda x: int(x.split('_')[-1]))
    
    print("=== Shenzhen Dataset Analysis ===")
    print(f"Total Shenzhen images: {len(sz_df)}")
    
    normal = sz_df[sz_df['true_label'] == 0]
    tb = sz_df[sz_df['true_label'] == 1]
    
    print(f"\nNormal Cases (_0): {len(normal)}")
    print(f"  Mean TB Prob: {normal['tb_prob'].mean():.4f}")
    print(f"  Mean Timika:  {normal['timika_score'].mean():.4f}")
    
    print(f"\nTB Cases (_1): {len(tb)}")
    print(f"  Mean TB Prob: {tb['tb_prob'].mean():.4f}")
    print(f"  Mean Timika:  {tb['timika_score'].mean():.4f}")

# Overall stats
print("\n=== Overall Analysis ===")
print(f"Total images: {len(df)}")
print(f"Mean TB Prob: {df['tb_prob'].mean():.4f}")
print(f"Min TB Prob:  {df['tb_prob'].min():.4f}")
print(f"Max TB Prob:  {df['tb_prob'].max():.4f}")
