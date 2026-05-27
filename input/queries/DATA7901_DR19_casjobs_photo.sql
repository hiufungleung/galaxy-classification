-- DATA7901 SDSS DR19 CasJobs query
-- Usage: Paste into SDSS CasJobs and run. Results will be saved to mydb.DATA7901_DR19

SELECT Distinct TOP (100) 
  p.objid,
  p.ra,
  p.dec,
  s.ra,
  s.dec,
  p.l,
  p.b,
  s.specObjID,
  s.plate,
  s.mjd,
  s.fiberid,
  s.programname,
  s.class,
  s.sdssPrimary,
  v.*,
  p.*
INTO mydb.DATA7901_DR19_photo
FROM PhotoObj AS p
  LEFT JOIN SpecObj AS s ON s.bestobjid = p.objid
  LEFT JOIN zooVotes AS v ON v.objid = p.objid
WHERE
  p.petroMag_r BETWEEN 10.0 AND 17.7
  AND s.zWarning = 0
  AND s.z BETWEEN 0.003333 AND 0.1500000001;


