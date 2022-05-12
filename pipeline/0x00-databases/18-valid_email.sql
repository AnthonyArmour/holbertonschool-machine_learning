-- resets valid_email when the email has been changed.
CREATE TRIGGER reset_validation
BEFORE UPDATE
ON users
FOR EACH ROW
	IF STRCMP(old.email, new.email) <> 0 THEN
	   SET new.valid_email = 0;
	END IF;
