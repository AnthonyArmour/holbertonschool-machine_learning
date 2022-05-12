-- Lists all shows contained in "hbtn_0d_tvshows" that have at least one genre linked
SELECT tv_shows.title, tv_genres.id AS genre_id FROM tv_show_genres
JOIN tv_shows ON tv_shows.id = tv_show_genres.show_id
JOIN tv_genres ON tv_genres.id = tv_show_genres.genre_id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
