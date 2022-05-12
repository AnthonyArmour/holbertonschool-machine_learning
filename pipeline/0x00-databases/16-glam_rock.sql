-- lists all bands with Glam rock as main style, ranked by longevity
SELECT band_name, IF(split IS NULL, (2020 - formed), (split - formed)) AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;