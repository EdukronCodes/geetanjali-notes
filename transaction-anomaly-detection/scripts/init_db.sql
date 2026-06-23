CREATE TABLE ledger_transactions (
    transaction_id VARCHAR(30) PRIMARY KEY,
    account_id VARCHAR(20) NOT NULL,
    vendor_id VARCHAR(20) NOT NULL,
    cost_center VARCHAR(30) NOT NULL,
    amount DECIMAL(18,2) NOT NULL,
    posting_date DATE NOT NULL,
    is_weekend BIT DEFAULT 0,
    analyst_label BIT NULL  -- 1 = true anomaly, 0 = normal, NULL = unlabeled
);

CREATE TABLE anomaly_runs (
    run_id INT IDENTITY(1,1) PRIMARY KEY,
    run_month VARCHAR(7) NOT NULL,
    total_transactions INT,
    flagged_count INT,
    precision_score FLOAT,
    output_file NVARCHAR(500),
    created_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX idx_ledger_posting_date ON ledger_transactions(posting_date);
CREATE INDEX idx_ledger_vendor ON ledger_transactions(vendor_id);
