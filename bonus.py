import pandas as pd
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CLASS CONNECTOR CỦA BẠN
# ==========================================
class Connector:
    def __init__(self,
                 server="localhost",
                 port=3306,
                 database="salesdatabase",
                 username="root",
                 password="@Obama123"):
        self.server = server
        self.port = port
        self.database = database
        self.username = username
        self.password = password

    def connect(self):
        self.conn = mysql.connector.connect(
            host=self.server,
            port=self.port,
            database=self.database,
            user=self.username,
            password=self.password,
            use_pure=True)

    def queryDataset(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df.columns = cursor.column_names
            return df
        except Exception as e:
            print(str(e))
            return None


# ==========================================
# 2. KẾT NỐI VÀ LẤY DỮ LIỆU TỪ SAKILA
# ==========================================
# Khởi tạo đối tượng Connector.
# Lưu ý: Đổi tham số database thành "sakila" và cập nhật mật khẩu nếu cần.
db = Connector(database="sakila", password="@Obama123")
db.connect()

sql_query = """
SELECT 
    c.customer_id,
    COUNT(r.rental_id) AS Total_Rentals,
    COUNT(DISTINCT i.film_id) AS Unique_Films_Rented,
    AVG(f.length) AS Avg_Film_Length_Preference,
    AVG(film_inv.total_inventory) AS Inventory_Availability_Preference
FROM customer c
JOIN rental r ON c.customer_id = r.customer_id
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN film f ON i.film_id = f.film_id
JOIN (
    SELECT film_id, COUNT(inventory_id) as total_inventory
    FROM inventory
    GROUP BY film_id
) film_inv ON f.film_id = film_inv.film_id
GROUP BY c.customer_id;
"""

print("Đang truy vấn dữ liệu từ database...")
df = db.queryDataset(sql_query)

if df is not None and not df.empty:
    print("Dữ liệu tải thành công. Kích thước:", df.shape)

    # Ép kiểu dữ liệu (tránh lỗi do MySQL trả về kiểu Decimal)
    df = df.astype(float)

    # ==========================================
    # 3. TIỀN XỬ LÝ VÀ GOM CỤM (K-MEANS)
    # ==========================================
    X = df.drop('customer_id', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    print("\nSố lượng khách hàng trong mỗi cụm:")
    print(df['Cluster'].value_counts())

    cluster_analysis = df.groupby('Cluster').mean().drop('customer_id', axis=1)
    print("\nĐặc trưng trung bình của từng cụm:")
    print(cluster_analysis)

    # ==========================================
    # 4. TRỰC QUAN HÓA (Đa biểu đồ)
    # ==========================================

    # --- Hình 1: Pairplot (Ma trận quan hệ giữa tất cả các biến) ---
    # Đổi tên cột cho ngắn gọn để dễ nhìn trên biểu đồ
    df_plot = df.rename(columns={
        'Total_Rentals': 'Rentals',
        'Unique_Films_Rented': 'Unique_Films',
        'Avg_Film_Length_Preference': 'Avg_Length',
        'Inventory_Availability_Preference': 'Inv_Availability'
    })

    # Chuyển Cluster sang kiểu category để seaborn đổi màu
    df_plot['Cluster'] = df_plot['Cluster'].astype('category')

    print("Đang vẽ biểu đồ Pairplot...")
    sns.pairplot(df_plot.drop('customer_id', axis=1), hue='Cluster', palette='viridis', corner=True)
    plt.suptitle('Ma trận tương quan các thuộc tính theo Cụm', y=1.02)
    plt.show()  # Hiển thị hình 1

    # --- Hình 2: Phân bố chi tiết của từng thuộc tính (Boxplot) ---
    print("Đang vẽ biểu đồ Boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.boxplot(ax=axes[0], x='Cluster', y='Total_Rentals', data=df, palette='Set2', hue='Cluster', legend=False)
    axes[0].set_title('Phân bố Tổng lượt thuê')

    sns.boxplot(ax=axes[1], x='Cluster', y='Avg_Film_Length_Preference', data=df, palette='Set2', hue='Cluster',
                legend=False)
    axes[1].set_title('Phân bố Thời lượng phim trung bình')

    sns.boxplot(ax=axes[2], x='Cluster', y='Inventory_Availability_Preference', data=df, palette='Set2', hue='Cluster',
                legend=False)
    axes[2].set_title('Mức độ phổ biến của phim')

    plt.tight_layout()
    plt.show()  # Hiển thị hình 2