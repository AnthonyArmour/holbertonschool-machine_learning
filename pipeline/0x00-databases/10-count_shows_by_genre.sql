-- lists all genres from hbtn_tvshows and displays number of shows linked to each
SELECT tv_genres.name AS genre, count(tv_show_genres.show_id) AS number_of_shows
FROM tv_genres INNER JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
GROUP BY tv_genres.name ORDER BY number_of_shows DESC;
