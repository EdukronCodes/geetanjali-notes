-- SQL Server schema for Finance Complaint Chatbot
CREATE TABLE accounts (
    account_id VARCHAR(20) PRIMARY KEY,
    customer_name NVARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at DATETIME2 DEFAULT GETDATE()
);

CREATE TABLE transactions (
    transaction_id VARCHAR(30) PRIMARY KEY,
    account_id VARCHAR(20) NOT NULL REFERENCES accounts(account_id),
    amount DECIMAL(18,2) NOT NULL,
    currency CHAR(3) DEFAULT 'USD',
    posting_date DATE NOT NULL,
    description NVARCHAR(255),
    status VARCHAR(20) DEFAULT 'POSTED',
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE complaints (
    complaint_id INT IDENTITY(1,1) PRIMARY KEY,
    account_id VARCHAR(20),
    transaction_id VARCHAR(30),
    intent VARCHAR(50),
    message NVARCHAR(MAX),
    bot_response NVARCHAR(MAX),
    escalated BIT DEFAULT 0,
    case_summary NVARCHAR(MAX),
    created_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX idx_transactions_account ON transactions(account_id);
CREATE INDEX idx_complaints_created ON complaints(created_at);
