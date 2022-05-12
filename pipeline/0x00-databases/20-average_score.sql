-- creates procedure that computes and stores the average score for a student.
CREATE PROCEDURE ComputeAverageScoreForUser (user_id INT)
BEGIN
    UPDATE users
    SET average_score = (
        SELECT AVG(corrections.score)
        FROM corrections WHERE corrections.user_id = user_id
        )
    WHERE id = user_id;
END;
