-- creates function that divides the first by the second number
CREATE FUNCTION SafeDiv (a INT, b INT)
RETURNS FLOAT
BEGIN
    IF b = 0 THEN
       RETURN 0;
    ELSE
            RETURN (a / b);
    END IF;
END
